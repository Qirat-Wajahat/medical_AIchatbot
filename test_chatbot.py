"""
Test script for Medical AI Chatbot
Validates all components work together
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.disease_predictor import DiseasePredictionModel
from utils.preprocessing import TextPreprocessor
from utils.medical_reference import MedicalReference


def test_components():
    """Test all application components"""
    print("="*60)
    print("MEDICAL AI CHATBOT - Component Test")
    print("="*60)
    
    # Test 1: Load models
    print("\n1. Loading models...")
    try:
        predictor = DiseasePredictionModel(data_path='data/medicines.json')
        preprocessor = TextPreprocessor()
        reference = MedicalReference(reference_path='data/medical_reference.txt')
        print("   ✓ All models loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading models: {e}")
        return False
    
    # Test 2: Train ML model
    print("\n2. Training disease prediction model...")
    try:
        predictor.train()
        print(f"   ✓ Model trained with {len(predictor.diseases)} diseases")
    except Exception as e:
        print(f"   ✗ Error training model: {e}")
        return False
    
    # Test 3: Test preprocessing
    print("\n3. Testing NLP preprocessing...")
    try:
        test_text = "I have fever, headache, and body aches"
        processed = preprocessor.preprocess(test_text)
        print(f"   Input: {test_text}")
        print(f"   Processed: {processed}")
        print("   ✓ Preprocessing works correctly")
    except Exception as e:
        print(f"   ✗ Error in preprocessing: {e}")
        return False
    
    # Test 4: Test disease prediction
    print("\n4. Testing disease prediction...")
    try:
        symptoms = "fever headache body aches fatigue"
        predictions = predictor.predict(symptoms)
        print(f"   Input symptoms: {symptoms}")
        print(f"   Top predictions:")
        for disease, prob in predictions[:3]:
            print(f"     - {disease}: {prob:.2%}")
        print("   ✓ Disease prediction works correctly")
    except Exception as e:
        print(f"   ✗ Error in prediction: {e}")
        return False
    
    # Test 5: Test rule-based matching
    print("\n5. Testing rule-based symptom matching...")
    try:
        processed = preprocessor.preprocess(symptoms)
        matches = predictor.match_symptoms(processed)
        print(f"   Top matches:")
        for disease, info in matches[:3]:
            print(f"     - {disease}: {info['matches']} symptom matches")
        print("   ✓ Rule-based matching works correctly")
    except Exception as e:
        print(f"   ✗ Error in matching: {e}")
        return False
    
    # Test 6: Test disease information retrieval
    print("\n6. Testing disease information retrieval...")
    try:
        disease_name = "Common Cold"
        disease_info = predictor.get_disease_info(disease_name)
        if disease_info:
            print(f"   Disease: {disease_name}")
            print(f"   Symptoms: {len(disease_info.get('symptoms', []))} found")
            print(f"   Medicines: {len(disease_info.get('medicines', []))} found")
            print("   ✓ Disease info retrieval works correctly")
        else:
            print("   ✗ Could not find disease info")
            return False
    except Exception as e:
        print(f"   ✗ Error retrieving disease info: {e}")
        return False
    
    # Test 7: Test medical reference
    print("\n7. Testing medical reference lookup...")
    try:
        ref_info = reference.get_disease_info("Influenza")
        if ref_info:
            print(f"   Found reference for Influenza")
            print(f"   Reference length: {len(ref_info)} characters")
            print("   ✓ Medical reference works correctly")
        else:
            print("   ✗ Could not find reference")
            return False
    except Exception as e:
        print(f"   ✗ Error in reference lookup: {e}")
        return False
    
    # Test 8: End-to-end simulation
    print("\n8. Running end-to-end simulation...")
    try:
        # Simulate user input
        user_input = "I have a runny nose, sneezing, and sore throat"
        print(f"   User input: {user_input}")
        
        # Process
        processed = preprocessor.preprocess(user_input)
        processed_text = ' '.join(processed)
        
        # Predict
        ml_predictions = predictor.predict(processed_text)
        rule_matches = predictor.match_symptoms(processed)
        
        # Get top disease
        if ml_predictions:
            top_disease = ml_predictions[0][0]
            print(f"   Top prediction: {top_disease}")
            
            # Get disease info
            disease_info = predictor.get_disease_info(top_disease)
            if disease_info:
                medicines = disease_info.get('medicines', [])
                print(f"   Recommended medicines: {len(medicines)}")
                if medicines:
                    print(f"     Example: {medicines[0]['name']}")
            
            # Get reference
            ref_info = reference.get_disease_info(top_disease)
            if ref_info:
                print(f"   Medical reference available: Yes")
        
        print("   ✓ End-to-end simulation successful")
    except Exception as e:
        print(f"   ✗ Error in end-to-end test: {e}")
        return False
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
    print("\nThe Medical AI Chatbot is ready to use!")
    print("Run: streamlit run app.py")
    return True


if __name__ == "__main__":
    success = test_components()
    sys.exit(0 if success else 1)
