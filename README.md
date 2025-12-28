# 🏥 Medical AI Chatbot

A Flask-based medical chatbot with a chat-style web UI. It collects symptoms over a few messages and recommends **one best-matching medicine per detected symptom group** (including image, dosage, and URL) from `data/medicines.json`.

## ⚠️ Medical Disclaimer

**IMPORTANT: This chatbot is for educational and informational purposes only.**

- This tool does NOT replace professional medical advice, diagnosis, or treatment.
- Always consult a qualified healthcare provider for medical concerns.
- In case of emergency, call your local emergency number immediately.
- The predictions are based on limited data and may not be accurate.
- Do not self-medicate based on this information.

## 🌟 Features

- **Chat UI (Flask + Templates)**: Bootstrap-based interface with a fixed bottom input bar
- **Multi-turn intake**: Asks for your name, then collects symptoms across turns (stored in Flask session)
- **Dataset-backed suggestions**: Uses `data/medicines.json` to recommend medicines (image + dosage + URL)
- **Follow-up questions (style-only)**: Uses `data/scenarios.txt` only to ask intake-style questions
- **Safety guardrails**: Avoids recommending antibiotics by default; includes prominent disclaimers

## 📁 Project Structure

```
medical_AIchatbot/
├── app_flask.py                    # New Flask web application (recommended)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── static/                          # Flask static assets
│   ├── logo.png                     # App logo (served at /static/logo.png)
│   └── img/                         # Additional images
├── templates/                      # Flask HTML templates
│   ├── base.html
│   └── index.html
├── data/
│   ├── medicines.json              # Medical knowledge base (symptoms/disease -> medicine + dosage + URL)
│   └── scenarios.txt               # Communication style only (follow-up questions)
├── models/
│   ├── __init__.py
│   └── disease_predictor.py        # Scenario follow-up retrieval (style-only)
└── utils/
   ├── __init__.py
   └── preprocessing.py            # NLP preprocessing utilities

Other:
- wsgi.py                           # WSGI entrypoint for production hosts
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Qirat-Wajahat/medical_AIchatbot.git
   cd medical_AIchatbot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Notes:
   - `requirements.txt` includes `gunicorn` for production deployments on Linux/macOS.
   - On Windows, `gunicorn` may not install/run. If you hit issues, install the core deps instead:
     ```powershell
     python -m pip install flask nltk
     ```

3. **Download NLTK data** (will be done automatically on first run):
   The application will automatically (best-effort) download required NLTK datasets (punkt, stopwords, wordnet).

   If your environment cannot download NLTK data, you can skip downloads by setting:
   - `DISABLE_NLTK_DOWNLOADS=1`

## 💻 Usage

### Running the Application

Run the Flask app (templates-based UI).

1. **Install dependencies** (if you haven't already):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

2. **(Recommended) Start the Flask app**:
```powershell
python app_flask.py
```

- Access: http://localhost:5000

3. **(Optional) Run the component tests**:
```powershell
python test_chatbot.py
```

4. **Use the chatbot**:
- Enter your symptoms in the input box and submit.
- Review likely conditions and medicine suggestions.
- **Always consult a healthcare professional** for diagnosis and treatment.

### Example Inputs

- "I have a runny nose, sneezing, and sore throat"
- "fever, body aches, fatigue, and headache"
- "severe headache with sensitivity to light and nausea"
- "diarrhea, vomiting, and stomach cramps"

## 🧠 How It Works

### 1. NLP Preprocessing
- Cleans and normalizes user input
- Tokenizes text into words
- Removes stopwords while preserving medical terms
- Lemmatizes tokens to base forms

### 2. Condition Inference
The app does not train an ML classifier. Instead, it:
- Detects broad symptom clusters (e.g., respiratory, GI, skin, urinary)
- Scores `data/medicines.json` items by token overlap + simple heuristics
- Returns **one best medicine per detected cluster** and avoids duplicates

### 3. Medicine Recommendation
- Retrieves medicines from `data/medicines.json`
- Shows **image, dosage, and URL** (when available)
- Includes a short “why this medicine” explanation

### 4. Follow-up questions (style-only)
- Uses `data/scenarios.txt` only as a source of intake-style follow-up questions.
- Scenario lines are sanitized to avoid diagnosis/treatment statements.

## 📊 What Conditions Are Supported?

Recommendations come from whatever entries exist in `data/medicines.json` (which may include many conditions and products). The app groups symptoms into broad clusters and selects a best-matching item for each detected cluster.

## 🛠️ Technology Stack

- **Backend**: Flask
- **Frontend**: Jinja templates + Bootstrap
- **NLP**: NLTK (with a lightweight fallback tokenizer if NLTK data isn’t available)
- **Production server**: Gunicorn (Linux/macOS)

## 📦 Dependencies

Dependencies are managed in `requirements.txt`:

```
flask>=2.0
nltk>=3.8
gunicorn>=21.2
```

## 🧯 Troubleshooting

### `ModuleNotFoundError: No module named 'flask'`
This usually means you’re running the app with a different Python than the one where you installed packages.

On Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app_flask.py
```

### NLTK Data Not Found

If you see an error about missing NLTK data, you can download it manually:

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

If your environment cannot download NLTK data, set `DISABLE_NLTK_DOWNLOADS=1` and the app will fall back to a lightweight tokenizer.

### Gunicorn on Windows

`gunicorn` is primarily for Linux/macOS production servers. On Windows, run the app with:
- `python app_flask.py`

## 🔮 Future Enhancements

- [ ] Expand disease database with more conditions
- [ ] Add multi-language support
- [ ] Implement deep learning models for better accuracy
- [ ] Add user authentication and history tracking
- [ ] Integrate with medical APIs for real-time data
- [ ] Add symptom severity assessment
- [ ] Implement chatbot conversation interface
- [ ] Add visualization of disease probability distributions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is for educational purposes only. Please ensure proper medical consultation before making any health-related decisions.

## 👤 Author

Qirat Wajahat

## 🙏 Acknowledgments

- Medical data compiled from publicly available health resources
- Built with open-source tools and libraries
- Inspired by the need for accessible health information

---

**Remember: This is an educational tool. Always consult qualified healthcare professionals for medical advice, diagnosis, and treatment.**
