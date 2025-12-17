from flask import Flask, render_template, request, session, jsonify
import os
import re
from models.disease_predictor import DiseasePredictionModel
from utils.preprocessing import TextPreprocessor
from utils.medical_reference import MedicalReference

# Explicitly set template and static folders to the project templates/static directories
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
static_dir = os.path.join(os.path.dirname(__file__), 'static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-me')

# Load models once on startup
predictor = DiseasePredictionModel(data_paths=['data/medicines.json', 'data/medicine_items_updated.json'])
if not predictor.load_model():
    predictor.train()
    predictor.save_model()

preprocessor = TextPreprocessor()
reference = MedicalReference(reference_path='data/medical_reference.txt')


def _is_greeting(text: str) -> bool:
    t = (text or '').strip().lower()
    if not t:
        return True
    return bool(re.fullmatch(r"(hi|hey|hello|hy|hii+|good\s*(morning|afternoon|evening))\b[!. ]*", t))


def _get_chat_history():
    return session.get('chat_history', [])


def _set_chat_history(history):
    session['chat_history'] = history[-30:]


def _append_message(role: str, text: str):
    history = _get_chat_history()
    history.append({'role': role, 'text': text})
    _set_chat_history(history)
    return history


def analyze_symptoms(text):
    tokens = preprocessor.preprocess(text)
    joined = ' '.join(tokens)
    ml_predictions = predictor.predict(joined)
    rule_matches = predictor.match_symptoms(tokens)

    # Combine top results
    all_predictions = {}
    for disease, prob in ml_predictions[:10]:
        all_predictions[disease] = {'ml_score': prob, 'match_score': 0}

    for disease, match_info in rule_matches[:10]:
        if disease in all_predictions:
            all_predictions[disease]['match_score'] = match_info['matches']
        else:
            all_predictions[disease] = {'ml_score': 0, 'match_score': match_info['matches']}

    top = sorted(all_predictions.items(), key=lambda x: (x[1]['match_score'], x[1]['ml_score']), reverse=True)[:5]

    results = []
    for disease, scores in top:
        info = predictor.get_disease_info(disease) or {}
        ref = reference.get_disease_info(disease) or ''
        results.append({
            'disease': disease,
            'ml_score': scores['ml_score'],
            'match_score': scores['match_score'],
            'symptoms': info.get('symptoms', []),
            'medicines': info.get('medicines', []),
            'reference': ref,
        })

    # Patient-facing response only (no confidence/stats)
    if not results:
        bot_message = (
            "I’m sorry you’re feeling unwell. I can’t name a likely condition from that alone.<br><br>"
            "<strong>Quick questions:</strong><br>"
            "1) How long have you had these symptoms?<br>"
            "2) What other symptoms do you have (cough/sore throat, vomiting/diarrhea, shortness of breath)?<br><br>"
            "<strong>Safety note:</strong> Seek medical care urgently if symptoms are severe or worsening."
        )
        return {'bot_message': bot_message}

    top_result = results[0]
    condition = top_result.get('disease', 'a common illness')
    meds = top_result.get('medicines', [])
    medicine_name = meds[0].get('name') if meds and isinstance(meds[0], dict) else None

    # Provide an OTC option when appropriate
    # If dataset medicine is missing, fall back to common OTC fever/pain relief when relevant keywords exist.
    otc = medicine_name
    if not otc:
        token_set = set(tokens)
        if {'fever', 'temperature', 'headache', 'pain', 'aches', 'ache'}.intersection(token_set):
            otc = 'acetaminophen (paracetamol)'

    questions = "1) How long have you had these symptoms? 2) Do you also have cough/sore throat, vomiting/diarrhea, or shortness of breath?"
    safety = "Seek medical care urgently if symptoms are severe or worsening."

    if otc:
        bot_message = (
            f"I’m sorry you’re feeling unwell.<br><br>"
            f"<strong>Likely condition:</strong> {condition}<br>"
            f"<strong>Common OTC option:</strong> {otc} (follow the label directions)<br><br>"
            f"<strong>Quick questions:</strong><br>"
            f"1) How long have you had these symptoms?<br>"
            f"2) Do you also have cough/sore throat, vomiting/diarrhea, or shortness of breath?<br><br>"
            f"<strong>Safety note:</strong> {safety}"
        )
    else:
        bot_message = (
            f"I’m sorry you’re feeling unwell.<br><br>"
            f"<strong>Likely condition:</strong> {condition}<br><br>"
            f"<strong>Quick questions:</strong><br>"
            f"1) How long have you had these symptoms?<br>"
            f"2) Do you also have cough/sore throat, vomiting/diarrhea, or shortness of breath?<br><br>"
            f"<strong>Safety note:</strong> {safety}"
        )

    return {'bot_message': bot_message}



@app.route('/', methods=['GET'])
def index():
    # Optional reset clears the multi-turn context
    if request.args.get('reset'):
        session.pop('chat_history', None)

    # Keep GET flow working (fallback if JS is disabled)
    query = (request.args.get('query', '') or '').strip()
    if query:
        _append_message('user', query)

        if _is_greeting(query):
            bot_message = (
                "Hi — I’m here to help.<br><br>"
                "Please tell me your symptoms (for example: fever, cough, sore throat, headache, nausea) and how long you’ve had them.<br><br>"
                "<strong>Safety note:</strong> If symptoms are severe or worsening, seek medical care urgently."
            )
        else:
            recent_user_texts = [m['text'] for m in _get_chat_history() if m.get('role') == 'user'][-5:]
            combined_text = ' '.join(recent_user_texts)
            bot_message = analyze_symptoms(combined_text).get('bot_message', '')

        _append_message('bot', bot_message)

    return render_template('index.html', chat_history=_get_chat_history())


@app.route('/chat', methods=['POST'])
def chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get('message') or '').strip()
    if not message:
        return jsonify({'ok': False, 'error': 'empty_message'}), 400

    _append_message('user', message)

    if _is_greeting(message):
        bot_message = (
            "Hi — I’m here to help.<br><br>"
            "Please tell me your symptoms (for example: fever, cough, sore throat, headache, nausea) and how long you’ve had them.<br><br>"
            "<strong>Safety note:</strong> If symptoms are severe or worsening, seek medical care urgently."
        )
    else:
        recent_user_texts = [m['text'] for m in _get_chat_history() if m.get('role') == 'user'][-5:]
        combined_text = ' '.join(recent_user_texts)
        bot_message = analyze_symptoms(combined_text).get('bot_message', '')

    _append_message('bot', bot_message)
    return jsonify({'ok': True, 'messages': _get_chat_history()})


if __name__ == '__main__':
    # Run without the debug reloader to avoid repeated heavy imports during development
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)
