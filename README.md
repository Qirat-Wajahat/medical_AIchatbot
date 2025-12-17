# 🏥 Medical AI Chatbot

An end-to-end AI-powered medical chatbot that analyzes patient symptoms using NLP and machine learning to predict possible diseases, recommend relevant medicines from real-world data, and provide educational medical information.

## ⚠️ Medical Disclaimer

**IMPORTANT: This chatbot is for educational and informational purposes only.**

- This tool does NOT replace professional medical advice, diagnosis, or treatment.
- Always consult a qualified healthcare provider for medical concerns.
- In case of emergency, call your local emergency number immediately.
- The predictions are based on limited data and may not be accurate.
- Do not self-medicate based on this information.

## 🌟 Features

- **Symptom Analysis**: Uses Natural Language Processing (NLP) to extract and process patient symptoms
- **Disease Prediction**: Machine Learning models predict possible diseases based on symptoms
- **Medicine Recommendations**: Suggests relevant medicines from a comprehensive JSON database
- **Medical Reference**: Displays educational medical information from reference texts
- **User-Friendly Interface**: Built with Streamlit for an intuitive user experience
- **Educational Disclaimer**: Includes prominent warnings about proper medical consultation

## 📁 Project Structure

```
medical_AIchatbot/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── data/
│   ├── medicines.json             # Disease and medicine database
│   └── medical_reference.txt      # Medical reference information
├── models/
│   ├── __init__.py
│   └── disease_predictor.py       # ML model for disease prediction
└── utils/
    ├── __init__.py
    ├── preprocessing.py           # NLP preprocessing utilities
    └── medical_reference.py       # Reference text handler
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

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the application**:
   - The app will open in your default browser
   - Default URL: `http://localhost:8501`

3. **Use the chatbot**:
   - Enter your symptoms in the text area (e.g., "fever, headache, body aches")
   - Click "Analyze Symptoms" button
   - Review the predicted diseases and recommendations
   - Read the medical reference information
   - **Always consult a healthcare professional**

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

### 2. Disease Prediction
- **ML Model**: Uses Multinomial Naive Bayes with TF-IDF features
- **Rule-Based Matching**: Matches symptoms with disease database
- **Hybrid Approach**: Combines both methods for better accuracy

### 3. Medicine Recommendation
- Retrieves medicines from JSON database
- Shows dosage and purpose for each medicine
- Provides comprehensive treatment information

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

```
streamlit==1.28.0
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
nltk==3.8.1
joblib==1.3.2
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
