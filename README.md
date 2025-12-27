# 🏥 Medical AI Chatbot

A Flask-based medical chatbot that analyzes user symptoms, suggests likely conditions, and recommends relevant medicines (with dosage + URL) from a JSON knowledge base.

## ⚠️ Medical Disclaimer

**IMPORTANT: This chatbot is for educational and informational purposes only.**

- This tool does NOT replace professional medical advice, diagnosis, or treatment.
- Always consult a qualified healthcare provider for medical concerns.
- In case of emergency, call your local emergency number immediately.
- The predictions are based on limited data and may not be accurate.
- Do not self-medicate based on this information.

## 🌟 Features

- **Symptom Analysis**: Lightweight NLP-style normalization for user symptom text
- **Condition Inference**: Matches symptoms against `data/medicines.json` (knowledge base)
- **Medicine Recommendations**: Suggests relevant medicines (dosage + URL) from the same JSON catalog
- **User-Friendly Interface**: Flask web UI (templates + static assets)
- **Educational Disclaimer**: Includes warnings about proper medical consultation

## 📁 Project Structure

```
medical_AIchatbot/
├── app_flask.py                    # New Flask web application (recommended)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── static/                          # Flask static assets
│   └── logo.png                     # App logo (served at /static/logo.png)
├── templates/                      # Flask HTML templates
│   ├── base.html
│   ├── home.html
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

3. **Download NLTK data** (will be done automatically on first run):
   The application will automatically download required NLTK datasets (punkt, stopwords, wordnet).

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
- Matches user-described symptoms to entries in `data/medicines.json`.
- **Hybrid Approach**: Combines both methods for better accuracy

### 3. Medicine Recommendation
- Retrieves medicines from `data/medicines.json`
- Shows dosage and purpose for each medicine
- Provides comprehensive treatment information

#### Optional: medicine images
- `data/medicine_items_updated.json` can be used as a best-effort lookup for medicine product images.
- Images may not always appear if the recommended medicine name doesn’t match a product name in the catalog.

### 4. Medical Reference
- Displays educational information about diseases
- Includes descriptions, duration, prevention, and when to see a doctor

## 📊 Supported Diseases

The system currently supports prediction for:
- Common Cold
- Influenza (Flu)
- Migraine
- Gastroenteritis
- Allergic Rhinitis
- Bronchitis
- Urinary Tract Infection
- Hypertension
- Type 2 Diabetes
- Anxiety Disorder

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **NLP**: NLTK (Natural Language Toolkit)
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib

## 📦 Dependencies

Dependencies are managed in `requirements.txt`. Key packages used by the project include:

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
nltk>=3.8
joblib>=1.2.0
flask>=2.0
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
