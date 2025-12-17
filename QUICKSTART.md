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

The required packages are:
- `streamlit` - Web application framework
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

### Start the Chatbot
```bash
streamlit run app.py
```

The application will start and automatically open in your default web browser at `http://localhost:8501`.

### Using the Chatbot

1. **Enter Symptoms**: In the text area, describe your symptoms in natural language
   - Example: "I have fever, headache, and body aches"
   - Be specific and include multiple symptoms if applicable

2. **Analyze**: Click the "🔍 Analyze Symptoms" button

3. **Review Results**: The chatbot will display:
   - Possible diseases ranked by probability
   - Associated symptoms for each disease
   - Recommended medicines with dosage information
   - Medical reference information
   - Confidence metrics

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

### ML Confidence
- Shows the machine learning model's confidence in the prediction
- Higher percentage indicates stronger confidence

### Symptom Matches
- Number of user symptoms that match the disease profile
- More matches suggest better alignment

### Medicine Recommendations
Each medicine card shows:
- Medicine name
- Recommended dosage
- Purpose/use

### Medical Reference
Educational information including:
- Disease description
- Typical duration
- Prevention tips
- When to see a doctor

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
If port 8501 is already in use:
```bash
streamlit run app.py --server.port 8502
```

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
Edit `app.py` to customize the user interface and styling.

## File Structure
```
medical_AIchatbot/
├── app.py                      # Main Streamlit application
├── test_chatbot.py            # Test script
├── requirements.txt           # Dependencies
├── data/
│   ├── medicines.json        # Disease and medicine database
│   └── medical_reference.txt # Medical reference text
├── models/
│   └── disease_predictor.py  # ML model
└── utils/
    ├── preprocessing.py      # NLP utilities
    └── medical_reference.py  # Reference handler
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
