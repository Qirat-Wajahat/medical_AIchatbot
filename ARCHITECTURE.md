# Medical AI Chatbot - Technical Architecture

## Overview

This document provides a detailed technical overview of the Medical AI Chatbot architecture, design decisions, and implementation details.

## Architecture

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    Flask App (app_flask.py)                         │
│  - Routes: GET / (renders chat), POST /chat (AJAX)                  │
│  - Session chat history (multi-turn)                                │
│  - Formats patient-facing reply (no internal stats shown)           │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
                               │
                   ┌───────────▼───────────┐
                   │   Templates (Jinja)   │
                   │ templates/base.html   │
                   │  - fixed bottom input │
                   │  - JS fetch('/chat')  │
                   │ templates/index.html  │
                   │  - chat board/bubbles │
                   └───────────┬───────────┘
                               │
                               │
        ┌──────────────────────▼───────────────────────┐
        │                 Core Pipeline                 │
        │                                               │
        │  NLP Preprocessing (utils/preprocessing.py)   │
        │   - cleaning/tokenization/lemmatization       │
        │                                               │
        │  Disease Predictor (models/disease_predictor) │
        │   - ML model (TF-IDF + Naive Bayes)           │
        │   - Rule-based symptom matching               │
        │   - Model persistence (joblib .pkl)           │
        │                                               │
        │  Medical Reference (utils/medical_reference)  │
        │   - reference text lookup                     │
        └──────────────────────┬───────────────────────┘
                               │
                     ┌─────────▼─────────┐
                     │    Data Sources    │
                     │  - medicines.json  │
                     │  - medicine_items_ │
                     │    updated.json    │
                     │  - medical_reference.txt │
                     └────────────────────┘
```

## Components

### 1. Flask Application (app_flask.py)

**Purpose**: Main web application and orchestration

**Key Features**:
- Chat-style UI with a fixed bottom input bar
- Multi-turn conversation using Flask session storage
- AJAX endpoint (`POST /chat`) for real-time chat without page reloads
- Patient-facing response formatting (hides internal model scores)

**Design Pattern**: MVC-ish
- View: Jinja templates + minimal client-side JS
- Controller: Flask routes (`/` and `/chat`)
- Model: ML predictor + reference lookup

**State Strategy**:
- Model is loaded once at process startup (and restored from `.pkl` if present)
- User chat history is stored per-browser-session (Flask session cookie)

### 2. Disease Prediction Model (models/disease_predictor.py)

**Purpose**: Machine learning-based disease prediction

**Components**:

#### ML Model
- **Algorithm**: Multinomial Naive Bayes
- **Reason**: Effective for text classification, handles multi-class problems well, fast training
- **Features**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Max Features**: 100 (balanced between accuracy and performance)

#### Rule-Based Matcher
- **Purpose**: Complement ML predictions with symptom matching
- **Algorithm**: Keyword matching with scoring
- **Scoring**: Based on number of matching symptoms and match percentage

#### Training Data Generation
- **Source**: JSON datasets under `data/` (e.g., `medicines.json`, `medicine_items_updated.json`)
- **Augmentation**: Creates partial symptom combinations for robustness
- **Strategy**: For each disease, generates multiple training examples with varying symptom subsets

**Trade-offs**:
- **Accuracy vs Speed**: Chose Naive Bayes for faster inference over more complex models
- **Data Size**: Limited dataset compensated by data augmentation and hybrid approach

### 3. NLP Preprocessing (utils/preprocessing.py)

**Purpose**: Text cleaning and normalization for consistent processing

**Pipeline**:
1. **Text Cleaning**: Lowercase, remove URLs, special characters
2. **Tokenization**: Split text into words using NLTK's punkt tokenizer
3. **Stopword Removal**: Remove common words but keep medical terms (pain, fever, etc.)
4. **Lemmatization**: Convert words to base form (e.g., "aches" → "ache")

**Medical Domain Adaptations**:
- Preserved medical stopwords: 'pain', 'fever', 'no', 'not', 'severe', 'mild', 'high', 'low'
- Reason: These words carry important medical meaning

**NLTK Dependencies**:
- punkt_tab: Tokenization
- stopwords: Common word filtering
- wordnet: Lemmatization dictionary

### 4. Medical Reference Handler (utils/medical_reference.py)

**Purpose**: Load and retrieve educational medical information

**Features**:
- Parse structured reference text
- Disease-specific information lookup
- Keyword search across all content

**Data Structure**:
- Text file with disease sections
- Each section contains: description, duration, prevention, when to see doctor

### 5. Data Sources

#### medicines.json
**Structure**:
```json
{
  "diseases": [
    {
      "name": "Disease Name",
      "symptoms": ["symptom1", "symptom2", ...],
      "medicines": [
        {
          "name": "Medicine Name",
          "dosage": "Dosage info",
          "purpose": "Purpose description"
        }
      ]
    }
  ]
}
```

**Content**:
- 10 common diseases
- 60+ symptoms total
- 30+ medicine recommendations
- Covers diverse medical conditions

#### medical_reference.txt
**Structure**: Plain text with section markers
**Content**: Educational information for each disease

## Machine Learning Details

### Model Training

**Training Data Generation**:
```python
# For each disease with N symptoms:
# 1. Full symptom set → 1 example
# 2. N partial sets (removing 1 symptom each) → N examples
# Total: 1 + N examples per disease
```

**Example**:
- Disease: Common Cold
- Symptoms: [runny nose, sneezing, cough, sore throat]
- Generates 5 training examples:
  1. "runny nose sneezing cough sore throat"
  2. "sneezing cough sore throat" (removed runny nose)
  3. "runny nose cough sore throat" (removed sneezing)
  4. "runny nose sneezing sore throat" (removed cough)
  5. "runny nose sneezing cough" (removed sore throat)

**Rationale**: Improves robustness when users don't mention all symptoms

### Hybrid Prediction Strategy

**Why Hybrid?**
- ML model: Good for partial matches and fuzzy matching
- Rule-based: Good for exact symptom matches
- Combined: Better overall accuracy and confidence

**Combination Logic**:
1. Get ML predictions with probabilities
2. Get rule-based matches with symptom counts
3. Rank by: (symptom matches, ML probability)
4. Display top 3 results

### Feature Engineering

**TF-IDF Vectorization**:
- Converts symptom text to numerical features
- Weighs importance of symptoms across diseases
- Max features: 100 (prevents overfitting with small dataset)

## Design Decisions

### Why Streamlit?
- **Rapid Development**: Quick prototyping of UI
- **Python Native**: No separate frontend stack needed
- **Interactive**: Built-in widgets and state management
- **Deployment**: Easy deployment options

### Why Naive Bayes?
- **Small Dataset**: Works well with limited training data
- **Fast**: Quick inference for real-time predictions
- **Interpretable**: Clear probability outputs
- **Text Classification**: Proven for text-based problems

### Why JSON for Data?
- **Human Readable**: Easy to edit and maintain
- **Structured**: Clear schema for diseases and medicines
- **No Database**: Simplifies deployment
- **Version Control**: Easy to track changes

### Why NLTK?
- **Comprehensive**: All needed NLP tools in one package
- **Mature**: Well-tested and documented
- **Lightweight**: Compared to spaCy or transformer models
- **Educational**: Good for learning NLP concepts

## Performance Considerations

### Model Loading
- Loaded at process startup
- If persisted model files exist, training is skipped
- Training occurs only when no saved model is available

### Prediction Speed
- ML prediction: < 100ms
- Rule-based matching: < 50ms
- Total inference: < 200ms

### Memory Usage
- Vectorizer: ~1MB
- Model: < 500KB
- Total: < 5MB (excluding libraries)

## Security Considerations

### Data Privacy
- No user data stored
- No external API calls
- All processing local

### Input Validation
- Text input only
- No database (SQL injection N/A)
- XSS: templates escape by default; bot message uses controlled HTML formatting

### Medical Disclaimer
- Prominently displayed
- Cannot be dismissed
- Clear warnings about limitations

## Scalability

### Current Limitations
- 10 diseases (expandable to 100+ without code changes)
- Single language (English)
- No user accounts or history

### Future Scalability
- Database: Replace JSON with SQL/NoSQL for larger datasets
- ML Model: Add more sophisticated models (RandomForest, Neural Networks)
- Deployment: Containerize with Docker for cloud deployment
- Multi-language: Add translation layer

## Testing Strategy

### Test Coverage (test_chatbot.py)
1. Model loading and initialization
2. ML model training
3. NLP preprocessing
4. Disease prediction
5. Rule-based matching
6. Information retrieval
7. Reference lookup
8. End-to-end workflow

### Testing Approach
- Unit tests for individual components
- Integration test for complete workflow
- Manual testing via Streamlit UI

## Deployment

### Local Deployment
```bash
python app_flask.py
```

### Production Deployment Options
1. **Streamlit Cloud**: Native hosting platform
2. **Heroku**: PaaS with buildpacks
3. **AWS/GCP/Azure**: VM or container-based
4. **Docker**: Containerized deployment

### Environment Requirements
- Python 3.8+
- 512MB RAM minimum
- No GPU required
- Internet (only for initial NLTK downloads)

## Monitoring and Maintenance

### Logs
- Streamlit logs: Model training status
- Console output: Debug information

### Maintenance Tasks
1. Update disease database (medicines.json)
2. Refresh medical references
3. Retrain model with new data
4. Update dependencies

### Known Limitations
- Limited to 10 diseases (design choice for MVP)
- English language only
- No personalization or user history
- Predictions based on limited training data

## Compliance

### Medical Device Regulations
- **Not a medical device**: Educational tool only
- **Clear disclaimers**: Multiple warnings throughout
- **No diagnostic claims**: Predictions framed as "possible conditions"

### Data Protection
- **No PII collected**: No user registration or data storage
- **GDPR compliant**: No data processing or storage
- **HIPAA N/A**: Not handling protected health information

## Conclusion

This architecture balances simplicity, functionality, and educational value. The hybrid ML-rule approach provides reasonable accuracy with limited data, while the comprehensive disclaimer system ensures responsible use.

The modular design allows for easy extension and maintenance, making it suitable as both a learning project and a foundation for more advanced medical AI systems.
