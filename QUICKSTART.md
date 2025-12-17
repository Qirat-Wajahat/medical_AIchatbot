# Quick Start Guide - Medical AI Chatbot

## Overview
This guide will help you get started with the Medical AI Chatbot application.

## Installation

### Step 1: Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)

### Step 2: Clone the Repository
```bash
git clone https://github.com/Qirat-Wajahat/medical_AIchatbot.git
cd medical_AIchatbot
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

The required packages include:
- `flask` - Web application framework (chat UI)
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning
- `nltk` - Natural language processing
- `joblib` - Model persistence

### Step 4: Verify Installation
Run the test script to ensure everything is working:
```bash
python test_chatbot.py
```

You should see "ALL TESTS PASSED! ✓" if the installation was successful.

## Running the Application

### Start the Chatbot (Flask)
```bash
python app_flask.py
```

Then open your browser at `http://127.0.0.1:5000`.

### Using the Chatbot

1. **Enter Symptoms**: Use the bottom chat bar to describe symptoms in natural language
   - Example: "I have fever, headache, and body aches"
   - Be specific and include multiple symptoms if applicable

2. **Continue the conversation**: Send follow-up messages (e.g., "and cough") and the app will keep chat history.

3. **Review Results**: The chatbot replies in a short patient-facing format:
   - A likely condition (when it can infer one)
   - A commonly recommended OTC medicine (when appropriate)
   - 1–2 follow-up questions
   - A brief safety note

4. **Important**: Always consult a healthcare professional for proper diagnosis and treatment

## Example Inputs

### Common Cold
```
runny nose, sneezing, and sore throat
```

### Flu
```
high fever, body aches, fatigue, and headache
```

### Migraine
```
severe headache with sensitivity to light and nausea
```

### Allergies
```
sneezing, itchy nose, and watery eyes
```

## Understanding the Results

The UI intentionally hides internal model details (confidence, ranking, etc.) and shows only the patient-facing reply.

## Important Disclaimers

⚠️ **This chatbot is for educational purposes only**

- Do NOT use this as a substitute for professional medical advice
- Always consult a qualified healthcare provider for diagnosis and treatment
- In emergencies, call your local emergency number immediately
- Do NOT self-medicate based on this information

## Troubleshooting

### NLTK Data Not Found
If you see an error about missing NLTK data:
```python
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Port Already in Use
If port 5000 is already in use, edit the port at the bottom of `app_flask.py` or run on a different port.

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Customization

### Adding New Diseases
Edit `data/medicines.json` to add new diseases with their symptoms and medicines.

### Updating Medical Reference
Edit `data/medical_reference.txt` to add or update medical information.

### Modifying the UI
Edit templates in `templates/` to customize the user interface and styling.

## File Structure
```
medical_AIchatbot/
├── app_flask.py                # Main Flask application
├── app.py                      # (Legacy) old Streamlit entry (deprecated)
├── test_chatbot.py            # Test script
├── requirements.txt           # Dependencies
├── data/
│   ├── medicines.json        # Disease and medicine database
│   ├── medicine_items_updated.json # Additional disease/medicine dataset
│   └── medical_reference.txt # Medical reference text
├── models/
│   └── disease_predictor.py  # ML model
│   ├── disease_model.pkl      # Saved model (created after first run)
│   └── vectorizer.pkl         # Saved vectorizer (created after first run)
└── utils/
    ├── preprocessing.py      # NLP utilities
    └── medical_reference.py  # Reference handler
templates/
├── base.html                  # Layout + bottom chat bar
└── index.html                 # Chat board
```

## Support

For issues or questions:
- Create an issue on GitHub
- Check the README.md for detailed documentation

## Next Steps

After successfully running the chatbot:
1. Try different symptom combinations
2. Explore the medical reference information
3. Review the code to understand how it works
4. Consider contributing improvements

---

**Remember**: This is an educational tool. Always seek professional medical advice for health concerns.
