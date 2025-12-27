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

On Windows (recommended), use a virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The required packages include:
- `flask` - Web application framework (chat UI)
- `nltk` - Natural language processing

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

### `ModuleNotFoundError: No module named 'flask'`
This usually means you’re running `python app_flask.py` with a different Python than the one where you installed packages.

Fix (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app_flask.py
```

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
Medical reference TXT is not used in this project.

### Modifying the UI
Edit templates in `templates/` to customize the user interface and styling.

## File Structure
```
medical_AIchatbot/
├── app_flask.py                # Main Flask application
├── test_chatbot.py            # Test script
├── requirements.txt           # Dependencies
├── static/
│   └── logo.png                # App logo (served at /static/logo.png)
├── data/
│   ├── medicines.json        # Disease and medicine database
│   └── scenarios.txt          # Communication style only
├── models/
│   └── disease_predictor.py  # Scenario follow-up retrieval (style-only)
└── utils/
    ├── preprocessing.py      # NLP utilities
templates/
├── base.html                  # Layout + bottom chat bar
└── index.html                 # Chat board
```

## Notes

### Data
The chatbot uses `data/medicines.json` for symptom/condition matching and medicine suggestions.

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
