"""
Medical AI Chatbot - Streamlit Application
End-to-end AI medical chatbot for symptom analysis and disease prediction
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.disease_predictor import DiseasePredictionModel
from utils.preprocessing import TextPreprocessor
from utils.medical_reference import MedicalReference


# Page configuration
st.set_page_config(
    page_title="Medical AI Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 20px 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        margin: 20px 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 15px;
        margin: 20px 0;
        border-radius: 5px;
    }
    .medicine-card {
        background-color: #f8f9fa;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load and cache models"""
    predictor = DiseasePredictionModel(data_path='data/medicines.json')
    preprocessor = TextPreprocessor()
    reference = MedicalReference(reference_path='data/medical_reference.txt')
    
    # Train the model
    predictor.train()
    
    return predictor, preprocessor, reference


def display_disclaimer():
    """Display medical disclaimer"""
    st.markdown("""
        <div class="disclaimer-box">
            <h3>⚠️ Medical Disclaimer</h3>
            <p><strong>IMPORTANT: This chatbot is for educational and informational purposes only.</strong></p>
            <ul>
                <li>This tool does NOT replace professional medical advice, diagnosis, or treatment.</li>
                <li>Always consult a qualified healthcare provider for medical concerns.</li>
                <li>In case of emergency, call your local emergency number immediately.</li>
                <li>The predictions are based on limited data and may not be accurate.</li>
                <li>Do not self-medicate based on this information.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)


def display_medicine_info(medicines):
    """Display medicine information in cards"""
    st.markdown("### 💊 Recommended Medicines")
    
    for idx, medicine in enumerate(medicines, 1):
        st.markdown(f"""
            <div class="medicine-card">
                <h4>{idx}. {medicine['name']}</h4>
                <p><strong>Dosage:</strong> {medicine['dosage']}</p>
                <p><strong>Purpose:</strong> {medicine['purpose']}</p>
            </div>
        """, unsafe_allow_html=True)


def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical AI Chatbot</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-Powered Symptom Analysis & Disease Prediction</p>',
        unsafe_allow_html=True
    )
    
    # Display disclaimer
    display_disclaimer()
    
    # Load models
    try:
        predictor, preprocessor, reference = load_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About This App")
        st.markdown("""
        This AI Medical Chatbot uses:
        - **Natural Language Processing (NLP)** for symptom analysis
        - **Machine Learning** for disease prediction
        - **Medical Database** for medicine recommendations
        
        **How to use:**
        1. Enter your symptoms in the text area
        2. Click "Analyze Symptoms"
        3. Review predictions and recommendations
        4. Consult a doctor for proper diagnosis
        """)
        
        st.markdown("---")
        st.markdown("### 🔍 Quick Tips")
        st.markdown("""
        - Be specific about your symptoms
        - Include duration and severity
        - Mention multiple symptoms if present
        - Example: "fever, headache, and sore throat"
        """)
    
    # Main content area
    st.markdown("## Enter Your Symptoms")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Symptom input
        symptoms_input = st.text_area(
            "Describe your symptoms:",
            height=150,
            placeholder="Example: I have a fever, headache, body aches, and fatigue...",
            help="Enter your symptoms in natural language. Be as detailed as possible."
        )
        
        # Analyze button
        analyze_button = st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Common Symptoms")
        st.markdown("""
        - Fever
        - Headache
        - Cough
        - Fatigue
        - Nausea
        - Pain
        - Dizziness
        - Shortness of breath
        """)
    
    # Process symptoms when button is clicked
    if analyze_button:
        if not symptoms_input.strip():
            st.warning("⚠️ Please enter your symptoms before analyzing.")
        else:
            with st.spinner("Analyzing your symptoms..."):
                # Preprocess symptoms
                processed_symptoms = preprocessor.preprocess(symptoms_input)
                symptoms_text = ' '.join(processed_symptoms)
                
                # Get predictions using ML model
                ml_predictions = predictor.predict(symptoms_text)
                
                # Get rule-based matches
                rule_based_matches = predictor.match_symptoms(processed_symptoms)
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Show preprocessed symptoms
                with st.expander("🔎 Processed Symptoms"):
                    st.info(f"Extracted keywords: {', '.join(processed_symptoms)}")
                
                # Display predictions
                if ml_predictions or rule_based_matches:
                    st.markdown("### 🎯 Possible Conditions")
                    
                    # Combine and rank predictions
                    all_predictions = {}
                    
                    # Add ML predictions
                    for disease, prob in ml_predictions[:5]:
                        all_predictions[disease] = {
                            'ml_score': prob,
                            'match_score': 0
                        }
                    
                    # Add rule-based matches
                    for disease, match_info in rule_based_matches[:5]:
                        if disease in all_predictions:
                            all_predictions[disease]['match_score'] = match_info['matches']
                        else:
                            all_predictions[disease] = {
                                'ml_score': 0,
                                'match_score': match_info['matches']
                            }
                    
                    # Display top predictions
                    top_diseases = sorted(
                        all_predictions.items(),
                        key=lambda x: (x[1]['match_score'], x[1]['ml_score']),
                        reverse=True
                    )[:3]
                    
                    for idx, (disease, scores) in enumerate(top_diseases, 1):
                        with st.expander(f"#{idx} {disease}", expanded=(idx == 1)):
                            # Get disease information
                            disease_info = predictor.get_disease_info(disease)
                            
                            if disease_info:
                                # Display symptoms
                                st.markdown("#### 🩺 Associated Symptoms")
                                symptoms_list = disease_info.get('symptoms', [])
                                st.write(", ".join(symptoms_list))
                                
                                # Display medicines
                                medicines = disease_info.get('medicines', [])
                                if medicines:
                                    display_medicine_info(medicines)
                                
                                # Display reference information
                                ref_info = reference.get_disease_info(disease)
                                if ref_info:
                                    st.markdown("#### 📚 Medical Reference")
                                    st.markdown(f"""
                                        <div class="info-box">
                                            {ref_info}
                                        </div>
                                    """, unsafe_allow_html=True)
                            
                            # Display confidence scores
                            st.markdown("#### 📈 Confidence Metrics")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if scores['ml_score'] > 0:
                                    st.metric("ML Confidence", f"{scores['ml_score']*100:.1f}%")
                            with col_b:
                                if scores['match_score'] > 0:
                                    st.metric("Symptom Matches", scores['match_score'])
                    
                    # Recommendations
                    st.markdown("---")
                    st.markdown("""
                        <div class="success-box">
                            <h3>💡 Next Steps</h3>
                            <ol>
                                <li><strong>Consult a Healthcare Professional:</strong> Get a proper diagnosis from a qualified doctor.</li>
                                <li><strong>Do Not Self-Medicate:</strong> Only take medicines prescribed by your doctor.</li>
                                <li><strong>Monitor Your Symptoms:</strong> Keep track of any changes in your condition.</li>
                                <li><strong>Emergency Care:</strong> If symptoms worsen or you experience severe distress, seek immediate medical attention.</li>
                            </ol>
                        </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.warning("⚠️ Could not match your symptoms with our database. Please consult a healthcare professional for proper evaluation.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>Medical AI Chatbot v1.0 | For Educational Purposes Only</p>
            <p>Always consult healthcare professionals for medical advice</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
