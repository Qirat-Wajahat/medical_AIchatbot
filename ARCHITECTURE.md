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
        │  Knowledge Matcher (app_flask.py)             │
        │   - Token overlap scoring against             │
        │     data/medicines.json entries               │
        │                                               │
        │  Style Followups (models/disease_predictor.py)│
        │   - Retrieves follow-up questions from        │
        │     data/scenarios.txt (style-only)           │
        │                                               │
        └──────────────────────┬───────────────────────┘
                               │
                     ┌─────────▼─────────┐
                     │    Data Sources    │
                     │  - medicines.json  │
                     │  - scenarios.txt   │
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
- Model: catalog matcher (medicines.json) + scenario follow-up retriever (style-only)

**State Strategy**:
- Data is loaded/cached once at process startup (JSON catalog + scenario samples)
- User chat history, stage, and short symptom history are stored per-browser-session (Flask session cookie)

### 2. Style Follow-ups (models/disease_predictor.py)

**Purpose**: Retrieve follow-up questions from `data/scenarios.txt`.

**Key rule**: Scenarios are used for communication style only (intake/follow-up questions).

**How it works**:
- Parses scenario blocks into patient-text + doctor-lines
- Ranks scenarios by token overlap with the user message
- Returns top follow-up lines (sanitized in `app_flask.py`)

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

### 4. Medical Knowledge Base (data/medicines.json)

**Purpose**: Primary medical knowledge source for:
- likely condition inference
- medicine recommendations (dosage + URL)

**Structure (high level)**:
- A JSON list of medicine/product-like objects
- Each object contains fields such as `name`, `@type`, `disease`, `symptoms`, `dosage`, `image`, and `url`

**Important app behavior**:
- Items without an `image` are skipped (the UI always shows a product image).
- The recommender avoids suggesting antibiotics by default.

### 5. Data Sources

#### scenarios.txt
**Purpose**: Communication style only.

**How it is used**:
- Used to source follow-up questions (e.g., duration, severity, red flags)
- Not used as medical knowledge

## Recommendation Logic (No ML Training)

This project intentionally does **not** train or persist a machine-learning classifier.

### High-level flow
1. **Preprocess input** with `TextPreprocessor` (clean/tokenize/remove stopwords/lemmatize; best-effort).
2. **Extract symptom keywords** from the known dataset and a small fallback list.
3. **Detect symptom clusters** (e.g., respiratory, GI, fever/pain, skin, urinary).
4. For each detected cluster, **score catalog items** by token overlap and a few heuristics:
   - prefer suitable forms for adult/child (tablet vs syrup)
   - prefer simpler dosing (once daily)
   - avoid antibiotics as default suggestions
5. **Pick one unique medicine per cluster**, then render a patient-facing reply.

## Design Decisions

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

### Startup
- `medicines.json` is loaded into an in-memory catalog cache.
- `scenarios.txt` is parsed into dialog samples for token-overlap retrieval.

### Request-time work
- The recommender uses lightweight token overlap and short heuristic scoring.
- Flask session is used to store chat history and a small rolling symptom buffer.

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
- English-language heuristics and tokenization
- No user accounts or long-term storage (session-only)
- Recommendation quality depends on the completeness/consistency of `medicines.json` fields (symptoms, images, dosage)

### Future Scalability
- Database: Replace JSON with SQL/NoSQL for larger datasets
- ML Model: Add more sophisticated models (RandomForest, Neural Networks)
- Deployment: Containerize with Docker for cloud deployment
- Multi-language: Add translation layer

## Testing Strategy

### Test Coverage (test_chatbot.py)
1. Component loading (predictor + preprocessor)
2. `medicines.json` load/parse
3. NLP preprocessing smoke test
4. Catalog-based recommendation function
5. Scenario follow-ups (style-only) + sanitization
6. End-to-end reply generation (`analyze_symptoms`)

### Testing Approach
- Unit tests for individual components
- Integration test for complete workflow
- Manual testing via the Flask web UI

## Deployment

### Local Deployment
```bash
python app_flask.py
```

### Production Deployment Options
1. **Gunicorn (Linux/macOS)**: Run the Flask app under a WSGI server
2. **PythonAnywhere / WSGI hosts**: Use `wsgi.py` as the entrypoint
3. **AWS/GCP/Azure**: VM or container-based deployment
4. **Docker**: Containerized deployment

### Environment Requirements
- Python 3.8+
- 512MB RAM minimum
- No GPU required
- Internet (only for initial NLTK downloads)

## Monitoring and Maintenance

### Logs
- Console output: Debug information

### Maintenance Tasks
1. Update disease database (medicines.json)
2. Update scenario prompts (scenarios.txt)
3. Update dependencies

### Known Limitations
- English language only
- No personalization or user history beyond a single browser session
- Recommendations depend on dataset quality (symptoms/images/dosage/URLs)

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

This architecture balances simplicity, functionality, and educational value. A lightweight dataset-matching approach keeps the app easy to run and modify, while the comprehensive disclaimer system encourages responsible use.

The modular design allows for easy extension and maintenance, making it suitable as both a learning project and a foundation for more advanced medical AI systems.
