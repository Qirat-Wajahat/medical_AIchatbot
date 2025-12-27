from flask import Flask, render_template, request, session, jsonify
import json
import os
import re
from models.disease_predictor import DiseasePredictionModel
from utils.preprocessing import TextPreprocessor

_BASE_DIR = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_BASE_DIR, 'data')
_MEDICINES_PATH = os.path.join(_DATA_DIR, 'medicines.json')
_SCENARIOS_PATH = os.path.join(_DATA_DIR, 'scenarios.txt')

_BOT_NAME = "Anna Balla"

# Define absolute paths for templates and static files
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
static_dir = os.path.join(os.path.dirname(__file__), 'static')

# Initialize Flask app with specified template and static folders
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# # Set secret key from environment variable with a safe development fallback
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-me')

# Load models once on startup
# `data/scenarios.txt` is used ONLY for communication style (questions/tone).
# `data/medicines.json` is used as the medical knowledge base (disease/symptoms -> medicines).
predictor = DiseasePredictionModel(
    data_paths=[_MEDICINES_PATH],
    scenario_path=_SCENARIOS_PATH,
    use_scenarios_for_training=False,
)

preprocessor = TextPreprocessor()

# medicines.json product catalog cache (contains disease/symptoms/dosage per item)
_MEDICINE_CATALOG = None

# Cache of symptom words (derived from known symptom phrases)
_SYMPTOM_WORDS = None


def _html_escape(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _format_welcome() -> str:
    return (
        f"<div><strong>Welcome</strong></div>"
        f"<div style=\"margin-top:0.35rem;\">Hello! I’m <strong>{_BOT_NAME}</strong>, your medical assistant. I can help you understand symptoms and suggest medicines from our dataset (educational only).</div>"
        f"<div style=\"margin-top:0.65rem;\"><strong>Name</strong></div>"
        f"<div style=\"margin-top:0.35rem;\">May I know your name?</div>"
        f"<div style=\"margin-top:0.65rem; color: rgba(255,255,255,0.78);\"><strong>Safety:</strong> If symptoms are severe or worsening, seek urgent medical care.</div>"
    )


def _extract_name(text: str) -> str | None:
    """Best-effort extraction of a person's name from a short message."""
    raw = (text or "").strip()
    if not raw:
        return None

    # Common patterns
    m = re.search(r"\b(?:my\s+name\s+is|i\s+am|i'm)\s+([A-Za-z][A-Za-z\-']{1,30}(?:\s+[A-Za-z][A-Za-z\-']{1,30})?)\b", raw, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
    else:
        candidate = raw

    candidate = re.sub(r"[^A-Za-z\-\s']", "", candidate).strip()
    candidate = re.sub(r"\s+", " ", candidate)
    if not candidate:
        return None

    # Keep it short and human-looking.
    parts = candidate.split()
    parts = parts[:2]
    candidate = " ".join(p.capitalize() for p in parts if p)
    if len(candidate) < 2:
        return None
    return candidate


def _ensure_welcome_in_history():
    """Ensure the very first bot message is the welcome + name prompt."""
    history = _get_chat_history()
    if not history:
        session['stage'] = 'awaiting_name'
        session.pop('user_name', None)
        session['symptom_history'] = []
        _append_message('bot', _format_welcome())
        return

    # If history exists but stage is missing (older sessions), default safely.
    if 'stage' not in session:
        session['stage'] = 'awaiting_name' if not session.get('user_name') else 'awaiting_symptoms'
    if 'symptom_history' not in session:
        session['symptom_history'] = []


def _load_medicine_catalog(path: str = _MEDICINES_PATH):
    """Load product catalog containing disease/symptoms/dosage fields."""
    global _MEDICINE_CATALOG
    if _MEDICINE_CATALOG is not None:
        return _MEDICINE_CATALOG

    try:
        with open(path, 'r', encoding='utf-8') as f:
            items = json.load(f)
    except FileNotFoundError:
        _MEDICINE_CATALOG = []
        return _MEDICINE_CATALOG
    except json.JSONDecodeError:
        _MEDICINE_CATALOG = []
        return _MEDICINE_CATALOG

    catalog = []
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            name = (it.get('name') or '').strip()
            if not name:
                continue
            med_type = (it.get('@type') or '').strip()
            disease = (it.get('disease') or '').strip()
            symptoms = (it.get('symptoms') or '').strip()
            dosage = (it.get('dosage') or '').strip()
            image = (it.get('image') or '').strip()
            url = (it.get('url') or '').strip()

            symptom_tokens = DiseasePredictionModel.normalize_text(symptoms).split() if symptoms else []
            disease_tokens = DiseasePredictionModel.normalize_text(disease).split() if disease else []
            blob = DiseasePredictionModel.normalize_text(' '.join([disease, symptoms, name]))

            catalog.append({
                'type': med_type,
                'name': name,
                'disease': disease,
                'symptoms': symptoms,
                'dosage': dosage,
                'image': image,
                'url': url,
                'symptom_tokens': symptom_tokens,
                'disease_tokens': disease_tokens,
                'blob': blob,
            })

    _MEDICINE_CATALOG = catalog
    return _MEDICINE_CATALOG


def _split_diseases(disease_field: str) -> list[str]:
    parts = []
    for p in (disease_field or '').split(','):
        p = p.strip()
        if p:
            parts.append(p)
    return parts


# Symptom clusters: group user symptoms into broader condition categories.
# This avoids treating each symptom as a separate "disease" and prevents duplicate medicines.
_SYMPTOM_CLUSTERS = [
    {
        'key': 'gastro',
        'label': 'Gastrointestinal upset (e.g., gastroenteritis)',
        'tokens': {
            'diarrhea', 'diarrhoea', 'vomit', 'vomiting', 'nausea', 'stomach', 'abdominal', 'cramp',
            'cramps', 'gastric', 'gastro', 'gastroenteritis', 'loose', 'stool', 'dehydration'
        },
    },
    {
        'key': 'respiratory',
        'label': 'Upper respiratory symptoms (e.g., cold/flu/allergies)',
        'tokens': {
            'cough', 'coughing', 'runny', 'nose', 'stuffy', 'congestion', 'sneeze', 'sneezing',
            'throat', 'sore', 'phlegm', 'cold', 'flu'
        },
    },
    {
        'key': 'pain_fever',
        'label': 'Fever / pain / headache',
        'tokens': {
            'fever', 'temperature', 'pain', 'ache', 'aches', 'headache', 'migraine', 'body', 'chills'
        },
    },
    {
        'key': 'skin',
        'label': 'Skin irritation (rash/itching)',
        'tokens': {
            'rash', 'itch', 'itching', 'redness', 'hives', 'eczema', 'acne', 'fungal', 'ringworm'
        },
    },
    {
        'key': 'urinary',
        'label': 'Urinary symptoms (possible UTI)',
        'tokens': {
            'burning', 'urination', 'urine', 'frequency', 'urgent', 'urgency', 'painful', 'uti'
        },
    },
]


def _detect_clusters(user_text: str):
    """Return clusters present in the user text, ranked by overlap."""
    user_tokens = _normalize_tokens(user_text)
    ranked = []
    for c in _SYMPTOM_CLUSTERS:
        overlap_tokens = user_tokens & c['tokens']
        if not overlap_tokens:
            continue
        ranked.append({
            'key': c['key'],
            'label': c['label'],
            'overlap_tokens': overlap_tokens,
            'overlap': len(overlap_tokens),
        })
    ranked.sort(key=lambda x: x['overlap'], reverse=True)
    return ranked


def _infer_patient_age_group(user_text: str) -> str:
    """Best-effort: returns 'child' or 'adult'."""
    t = (user_text or '').lower()
    if re.search(r"\b(child|kid|baby|toddler|infant|son|daughter)\b", t):
        return 'child'
    m = re.search(r"\b(\d{1,2})\s*(?:yo|y/o|years?\s*old)\b", t)
    if m:
        try:
            age = int(m.group(1))
            if age <= 12:
                return 'child'
        except ValueError:
            pass
    return 'adult'


def _normalize_tokens(text: str) -> set[str]:
    return set(DiseasePredictionModel.normalize_text(text).split())


def _normalize_form(med_type: str, name: str) -> str:
    t = (med_type or '').strip().lower()
    n = (name or '').strip().lower()
    # Prefer explicit @type, but also infer from name.
    if 'tablet' in t or 'tablets' in t or 'tablet' in n or 'tablets' in n:
        return 'tablet'
    if 'capsule' in t or 'capsules' in t or 'capsule' in n or 'capsules' in n or 'cap' in t:
        return 'capsule'
    if 'syrup' in t or 'syrup' in n:
        return 'syrup'
    if 'suspension' in t or 'suspension' in n:
        return 'suspension'
    if 'drops' in t or 'drop' in n:
        return 'drops'
    if 'lotion' in t or 'lotion' in n:
        return 'lotion'
    if 'cream' in t or 'cream' in n:
        return 'cream'
    if 'ointment' in t or 'ointment' in n:
        return 'ointment'
    if 'gel' in t or 'gel' in n:
        return 'gel'
    if 'liquid' in t or 'liquid' in n:
        return 'liquid'
    return (t or 'unknown')


def _medicine_unique_key(med: dict) -> str:
    """Return a stable key for de-duplicating medicines across clusters.

    We intentionally do NOT include URL here because the dataset can contain the
    same product multiple times with different/missing URLs.
    """
    name = (med or {}).get('name') or ''
    name = name.strip().lower()
    # Keep letters and digits (so strengths like 10mg vs 5mg remain distinct),
    # but normalize punctuation/whitespace.
    name = re.sub(r"[^a-z0-9]+", " ", name)
    name = " ".join(name.split())
    return name


def _dosage_simplicity(dosage: str) -> tuple[float, str | None]:
    """Returns (bonus_score, short_label)."""
    d = (dosage or '').lower()
    if not d:
        return 0.0, None

    # Strong preference: once daily.
    if re.search(r"\b(once\s*(a\s*)?day|once\s*daily|od|1\s*(tablet|tab)\s*daily|1\s*(tablet|tab)\s*once\s*daily)\b", d):
        return 1.5, 'once daily'

    # Next best: twice daily.
    if re.search(r"\b(twice\s*(a\s*)?day|twice\s*daily|bid|2\s*times\s*(a\s*)?day)\b", d):
        return 0.6, 'twice daily'

    # Penalize frequent dosing.
    if re.search(r"\b(three\s*times|3\s*times|every\s*\d+\s*(hours?|hrs?)|q\d+h)\b", d):
        return -0.4, 'multiple times daily'

    return 0.0, None


def _category_boost(user_tokens: set[str], med: dict) -> tuple[float, list[str]]:
    """Heuristics to reduce obviously-irrelevant picks.

    Goal: if the user mainly reports cough, prefer cough/cold products over
    allergy-only antihistamines unless allergy indicators are present.
    """
    blob = ((med or {}).get('blob') or '')
    name = ((med or {}).get('name') or '')
    med_text = f"{name} {blob}".lower()

    cough_indicators = {'cough', 'coughing', 'phlegm', 'sputum'}
    allergy_indicators = {'sneeze', 'sneezing', 'runny', 'nose', 'itch', 'itching', 'hives', 'allergy', 'watery', 'eyes'}

    has_cough = bool(user_tokens & cough_indicators)
    has_allergy = bool(user_tokens & allergy_indicators)

    antihistamine_markers = {
        'cetirizine', 'levocetirizine', 'loratadine', 'desloratadine', 'fexofenadine',
        'rigix', 'zyrtec', 'claritin', 'telfast'
    }
    cough_markers = {
        'cough', 'expectorant', 'antitussive', 'guaifenesin', 'dextromethorphan',
        'ambroxol', 'bromhexine'
    }

    is_antihistamine = any(m in med_text for m in antihistamine_markers)
    is_cough_product = any(m in med_text for m in cough_markers)

    score = 0.0
    why = []

    if has_cough:
        if is_cough_product:
            score += 0.9
            why.append('Cough-focused product')
        if is_antihistamine and not has_allergy:
            score -= 0.7
            why.append('Less suitable without allergy symptoms')

    if has_allergy and is_antihistamine:
        score += 0.7
        why.append('Fits allergy indicators (runny nose/sneezing/itching)')

    # Avoid pushing antibiotics as a default OTC suggestion.
    antibiotic_markers = {
        'amoxicillin', 'amoxycillin', 'azithromycin', 'ciprofloxacin', 'cefixime',
        'ceftriaxone', 'metronidazole', 'doxycycline', 'clavulanate'
    }
    if any(m in med_text for m in antibiotic_markers):
        score -= 1.0
        why.append('Prescription antibiotic (not OTC by default)')

    return score, why


def _is_antibiotic_product(med: dict) -> bool:
    """Hard safety guard: do not recommend antibiotics as default suggestions."""
    blob = ((med or {}).get('blob') or '')
    name = ((med or {}).get('name') or '')
    med_text = f"{name} {blob}".lower()
    antibiotic_markers = {
        'amoxicillin', 'amoxycillin', 'azithromycin', 'ciprofloxacin', 'cefixime',
        'ceftriaxone', 'metronidazole', 'doxycycline', 'clavulanate',
        'antibiotic', 'cephalosporin',
    }
    return any(m in med_text for m in antibiotic_markers)


def _recommend_one_medicine_per_cluster(user_text: str, max_clusters: int = 3):
    """Detect symptom clusters, then recommend ONE medicine per cluster.

    Also enforces uniqueness so the same medicine is not recommended multiple times.

    Returns:
        list[dict]: [{cluster_label, medicine, why(list[str])}]
    """
    catalog = _load_medicine_catalog()
    if not catalog:
        return []

    user_tokens = _normalize_tokens(user_text)
    if not user_tokens:
        return []

    age_group = _infer_patient_age_group(user_text)
    clusters = _detect_clusters(user_text)
    if not clusters:
        return []

    # Build candidate lists per cluster.
    skin_tokens = next((c['tokens'] for c in _SYMPTOM_CLUSTERS if c['key'] == 'skin'), set())

    candidates_by_cluster = {}
    for c in clusters:
        c_tokens = next((cc['tokens'] for cc in _SYMPTOM_CLUSTERS if cc['key'] == c['key']), set())
        cand = []
        best_by_key = {}
        for it in catalog:
            # Requirement: ALWAYS include medicine images.
            if not (it.get('image') or '').strip():
                continue

            # Safety: do not recommend antibiotics by default.
            if _is_antibiotic_product(it):
                continue

            symptom_tokens = set(it.get('symptom_tokens') or [])
            disease_tokens = set(it.get('disease_tokens') or [])
            blob_tokens = set((it.get('blob') or '').split())

            # Must match BOTH: user symptoms and cluster relevance.
            symptom_match = user_tokens & symptom_tokens
            if not symptom_match:
                continue
            cluster_match = symptom_match & c_tokens
            if not cluster_match:
                continue

            disease_match = user_tokens & disease_tokens
            blob_overlap = len(user_tokens & blob_tokens)

            base_score = (4.0 * len(cluster_match)) + (1.0 * len(symptom_match)) + (0.5 * len(disease_match)) + (0.25 * blob_overlap)

            form = _normalize_form(it.get('type') or '', it.get('name') or '')
            form_bonus = 0.0
            if age_group == 'adult':
                if form in {'tablet', 'capsule'}:
                    form_bonus += 1.2
                if form in {'syrup', 'suspension', 'drops', 'liquid'}:
                    form_bonus -= 0.2
            else:
                if form in {'syrup', 'suspension', 'drops', 'liquid'}:
                    form_bonus += 1.2
                if form in {'tablet', 'capsule'}:
                    form_bonus -= 1.0

            # Skin products should only be chosen for skin cluster.
            if form in {'cream', 'ointment', 'lotion', 'gel'}:
                if c['key'] != 'skin':
                    form_bonus -= 1.5
                elif not (user_tokens & skin_tokens):
                    form_bonus -= 0.5

            dosage_bonus, dosage_label = _dosage_simplicity(it.get('dosage') or '')

            category_bonus, category_why = _category_boost(user_tokens, it)

            total_score = base_score + form_bonus + dosage_bonus + category_bonus
            if total_score <= 0:
                continue

            matched_preview = sorted(list(cluster_match))[:4]
            why = []
            if matched_preview:
                why.append("Matches your symptoms: " + ', '.join(matched_preview))
            if age_group == 'child' and form in {'syrup', 'suspension', 'drops', 'liquid'}:
                why.append(f"Preferred form for a child: {form}")
            elif age_group == 'adult' and form in {'tablet', 'capsule'}:
                why.append(f"Preferred adult form: {form}")
            elif form != 'unknown':
                why.append(f"Form suitability: {form}")
            if dosage_label:
                why.append(f"Simple dosing: {dosage_label}")
            if category_why:
                why.extend(category_why[:2])

            entry = {'score': total_score, 'medicine': it, 'why': why}
            med_key = _medicine_unique_key(it)
            if not med_key:
                continue
            prev = best_by_key.get(med_key)
            if (prev is None) or (entry['score'] > prev['score']):
                best_by_key[med_key] = entry

        cand = list(best_by_key.values())

        cand.sort(key=lambda x: x['score'], reverse=True)
        candidates_by_cluster[c['key']] = cand

    # Greedy assignment: pick best unique medicine per cluster.
    used_meds = set()
    picks = []
    for c in clusters:
        if len(picks) >= max(1, max_clusters):
            break
        for cand in candidates_by_cluster.get(c['key'], []):
            med = cand['medicine']
            med_id = _medicine_unique_key(med)
            if not med_id:
                continue
            if med_id in used_meds:
                continue
            used_meds.add(med_id)
            picks.append({
                'cluster_label': c['label'],
                'medicine': med,
                'why': cand.get('why') or [],
                'score': cand.get('score', 0),
            })
            break

    return picks


def _sanitize_style_line(line: str) -> str | None:
    """Keep only communication-style questions; drop diagnosis/treatment statements."""
    t = (line or '').strip()
    if not t:
        return None

    lower = t.lower()

    # Never allow scenario hardcoded names to leak into the chat.
    if re.search(r"\b(okay|ok)[,\s]+[A-Z][a-z]+\s+[A-Z][a-z]+\b", t, flags=re.IGNORECASE):
        return None
    if 'imran ali' in lower:
        return None

    # Avoid re-asking for the name (we manage name collection explicitly).
    if re.search(r"\b(my\s+name\s+is|may\s+i\s+know\s+your\s+name|what\s+is\s+your\s+name)\b", lower):
        return None
    # Drop explicit diagnosis/conclusion lines
    banned_phrases = [
        'most likely represents',
        'this most likely represents',
        'given your symptoms',
        'this looks like',
        'these features are consistent',
        'we need blood tests',
        'confirm',
        'diagnosis',
        'treatment',
        'antibiotic',
        'prescribe',
    ]
    if any(p in lower for p in banned_phrases):
        return None

    # Keep mostly questions / intake prompts.
    if '?' in t:
        return t

    # Allow a few generic intake sentences
    allowed_starts = (
        'please', 'may i', 'okay', 'how can i', 'what brings you', 'do you have',
        'have you', 'any', 'is there', 'are you',
    )
    if lower.startswith(allowed_starts):
        return t

    return None


def _render_style_followups(user_text: str, limit: int = 3) -> str:
    """Use scenarios.txt only for question style (sanitized)."""
    lines = predictor.get_scenario_followups(user_text, top_k=max(6, limit * 2))
    cleaned = []
    for ln in lines:
        s = _sanitize_style_line(ln)
        if s and s not in cleaned:
            cleaned.append(s)
        if len(cleaned) >= limit:
            break

    if not cleaned:
        return ''

    items = ''.join(f"<li>{ln}</li>" for ln in cleaned)
    return (
        "<div style=\"margin-top: 0.5rem;\">"
        "<strong>Quick questions (style from scenarios):</strong>"
        "<ul style=\"margin: 0.35rem 0 0.5rem 1.25rem;\">"
        f"{items}"
        "</ul>"
        "</div>"
    )


def _get_symptom_words():
    global _SYMPTOM_WORDS
    if _SYMPTOM_WORDS is not None:
        return _SYMPTOM_WORDS

    words = set()

    # Also learn symptom words from medicines.json catalog.
    for item in _load_medicine_catalog():
        for w in (item.get('symptom_tokens') or []):
            words.add(w)

    # Small fallback set to keep the bot responsive even with sparse datasets.
    words.update({
        'fever', 'temperature', 'headache', 'pain', 'ache', 'aches', 'nausea', 'vomit', 'vomiting',
        'diarrhea', 'diarrhoea', 'cough', 'sore', 'throat', 'runny', 'nose', 'congestion',
        'burning', 'urination', 'rash', 'itching', 'redness', 'fatigue', 'flu'
    })
    _SYMPTOM_WORDS = words
    return _SYMPTOM_WORDS


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


def analyze_symptoms(text, user_name: str | None = None):
    name_prefix = (f"{_html_escape(user_name)}, " if user_name else "")

    tokens = preprocessor.preprocess(text)
    symptom_words = _get_symptom_words()

    # Use lightweight normalized tokens for matching against medicines.json strings.
    norm_tokens = DiseasePredictionModel.normalize_text(text).split()
    relevant_tokens = [t for t in norm_tokens if t in symptom_words]
    unique_relevant_count = len(set(relevant_tokens))

    # Some symptoms are specific enough to allow suggestions even if only one is present.
    high_signal_single_symptoms = {
        'diarrhea', 'diarrhoea', 'vomit', 'vomiting', 'nausea',
        'burning', 'urination', 'uti',
        'rash', 'itch', 'itching'
    }
    has_high_signal = bool(set(relevant_tokens) & high_signal_single_symptoms)

    recommendations = _recommend_one_medicine_per_cluster(text, max_clusters=3)

    # If the user only provided 0–1 symptom keywords, avoid guessing (common cause of wrong meds).
    # Exception: allow high-signal single symptoms (e.g., diarrhea, vomiting, UTI-like burning).
    if unique_relevant_count < 2 and not has_high_signal:
        otc = None
        token_set = set(tokens)
        if {'fever', 'temperature', 'headache', 'pain', 'aches', 'ache'}.intersection(token_set):
            otc = 'acetaminophen (paracetamol)'

        bot_message = (
            f"<div><strong>Symptoms</strong></div>"
            f"<div style=\"margin-top:0.35rem;\">{name_prefix}I’m sorry you’re feeling unwell. I don’t have enough detail to name a likely condition yet.</div><br>"
            + (f"<strong>Common OTC option:</strong> {otc} (follow the label directions)<br><br>" if otc else "")
            + "<strong>Quick questions:</strong><br>"
            + "1) How long have you had these symptoms?<br>"
            + "2) What are the top 3 symptoms (for example: cough/sore throat, vomiting/diarrhea, burning urination, headache)?<br>"
            + "3) Any emergency warning signs (chest pain, trouble breathing, confusion, blood in vomit/stool/urine)?<br><br>"
            + "<strong>Safety note:</strong> Seek medical care urgently if symptoms are severe or worsening."
            + _render_style_followups(text, limit=3)
        )
        return {'bot_message': bot_message, 'had_recommendations': False}

    # Use medicines.json as the medical knowledge base.
    # Patient-facing response only (no confidence/stats)
    if not recommendations:
        bot_message = (
            f"<div><strong>Symptoms</strong></div>"
            f"<div style=\"margin-top:0.35rem;\">{name_prefix}I’m sorry you’re feeling unwell. I can’t name a likely condition from that alone.</div><br>"
            "<strong>Quick questions:</strong><br>"
            "1) How long have you had these symptoms?<br>"
            "2) What other symptoms do you have (cough/sore throat, vomiting/diarrhea, shortness of breath)?<br><br>"
            "<strong>Safety note:</strong> Seek medical care urgently if symptoms are severe or worsening."
            + _render_style_followups(text, limit=3)
        )
        return {'bot_message': bot_message, 'had_recommendations': False}

    # Render concise: one medicine per detected disease.
    rec_lines = []
    for rec in recommendations:
        disease = rec.get('cluster_label') or 'Possible condition'
        med = rec.get('medicine') or {}
        med_name = (med.get('name') or '').strip() or 'a suitable medicine'
        dosage = (med.get('dosage') or '').strip()
        url = (med.get('url') or '').strip()
        img = (med.get('image') or '').strip()

        # Image is required by design; selection filters items without image.
        img_html = (
            f"<div style=\"margin-top:0.35rem;\">"
            f"<img src=\"{img}\" alt=\"{_html_escape(med_name)}\" "
            f"style=\"max-width:160px; width:160px; height:auto; border-radius:10px; border:1px solid rgba(255,255,255,0.06);\">"
            f"</div>"
        )

        parts = [f"<strong>{_html_escape(disease)}:</strong> {med_name}"]
        if dosage:
            parts.append(f" — <em>Dosage:</em> {dosage}")
        if url:
            parts.append(f" <a href=\"{url}\" target=\"_blank\" rel=\"noopener noreferrer\">View</a>")

        why = rec.get('why') or []
        why_html = ''
        if why:
            why_html = (
                "<div style=\"margin-top: 0.25rem; color: rgba(255,255,255,0.78);\">"
                "<em>Why this medicine:</em> " + "; ".join(why[:3]) +
                "</div>"
            )

        rec_lines.append(
            "<li style=\"margin-bottom: 0.75rem;\">" + ''.join(parts) + img_html + why_html + "</li>"
        )

    medicines_html = (
        "<div><strong>Recommendations</strong></div>"
        "<div style=\"margin-top:0.35rem;\">One best medicine per detected condition (not an exhaustive list).</div>"
        "<ul style=\"margin: 0.5rem 0 0.75rem 1.25rem;\">"
        + ''.join(rec_lines)
        + "</ul>"
        "<div style=\"color: rgba(255,255,255,0.75); font-size: 0.95em;\">"
        "Educational only — always follow the label and consult a clinician if unsure."
        "</div>"
    )

    safety = "Seek medical care urgently if symptoms are severe or worsening."

    bot_message = (
        f"<div><strong>Summary</strong></div>"
        f"<div style=\"margin-top:0.35rem;\">{name_prefix}here’s what I found from your symptoms:</div><br>"
        + medicines_html
        + "<br>"
        + _render_style_followups(text, limit=3)
        + f"<div style=\"margin-top:0.65rem;\"><strong>Safety note:</strong> {safety}</div>"
    )

    return {'bot_message': bot_message, 'had_recommendations': True}



@app.route('/', methods=['GET'])
def index():
    # Optional reset clears the multi-turn context
    if request.args.get('reset'):
        session.pop('chat_history', None)
        session.pop('user_name', None)
        session.pop('stage', None)
        session.pop('symptom_history', None)

    _ensure_welcome_in_history()

    # Used by the UI as a fallback in case chat history is empty for any reason.
    welcome_message = _format_welcome()

    # Keep GET flow working (fallback if JS is disabled)
    query = (request.args.get('query', '') or '').strip()
    if query:
        _append_message('user', query)

        stage = session.get('stage') or 'awaiting_name'

        if stage == 'awaiting_name':
            name = _extract_name(query)
            if not name:
                bot_message = (
                    "I didn’t catch your name. May I know your name?"
                )
            else:
                session['user_name'] = name
                session['stage'] = 'awaiting_symptoms'
                bot_message = (
                    f"Nice to meet you, <strong>{_html_escape(name)}</strong>!<br><br>"
                    "<div><strong>Symptoms</strong></div>"
                    "<div style=\"margin-top:0.35rem;\">How can I help today? Please describe your symptoms (and how long you’ve had them).</div>"
                )
            _append_message('bot', bot_message)
        else:
            symptom_history = session.get('symptom_history') or []
            symptom_history.append(query)
            session['symptom_history'] = symptom_history[-5:]
            combined_text = ' '.join(session['symptom_history'])
            analysis = analyze_symptoms(combined_text, user_name=session.get('user_name'))
            bot_message = analysis.get('bot_message', '')
            if analysis.get('had_recommendations'):
                session['symptom_history'] = []
            _append_message('bot', bot_message)

    return render_template('index.html', chat_history=_get_chat_history(), welcome_message=welcome_message)


@app.route('/chat', methods=['POST'])
def chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get('message') or '').strip()
    if not message:
        return jsonify({'ok': False, 'error': 'empty_message'}), 400

    _ensure_welcome_in_history()
    _append_message('user', message)

    stage = session.get('stage') or 'awaiting_name'

    if stage == 'awaiting_name':
        name = _extract_name(message)
        if not name:
            bot_message = "I didn’t catch your name. May I know your name?"
        else:
            session['user_name'] = name
            session['stage'] = 'awaiting_symptoms'
            bot_message = (
                f"Nice to meet you, <strong>{_html_escape(name)}</strong>!<br><br>"
                "<div><strong>Symptoms</strong></div>"
                "<div style=\"margin-top:0.35rem;\">How can I help today? Please describe your symptoms (and how long you’ve had them).</div>"
            )
        _append_message('bot', bot_message)
        return jsonify({'ok': True, 'messages': _get_chat_history()})

    # awaiting_symptoms
    symptom_history = session.get('symptom_history') or []
    symptom_history.append(message)
    session['symptom_history'] = symptom_history[-5:]
    combined_text = ' '.join(session['symptom_history'])
    analysis = analyze_symptoms(combined_text, user_name=session.get('user_name'))
    bot_message = analysis.get('bot_message', '')
    if analysis.get('had_recommendations'):
        session['symptom_history'] = []
    _append_message('bot', bot_message)
    return jsonify({'ok': True, 'messages': _get_chat_history()})


if __name__ == '__main__':
    # Run without the debug reloader to avoid repeated heavy imports during development
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)
