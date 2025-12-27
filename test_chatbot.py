"""Test script for Medical AI Chatbot.

This project uses:
- scenarios.txt for communication style only
- medicines.json as the medical knowledge base (symptoms/disease -> medicine suggestions)
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.disease_predictor import DiseasePredictionModel
from utils.preprocessing import TextPreprocessor
import json


def test_components():
    """Test all application components"""
    print("="*60)
    print("MEDICAL AI CHATBOT - Component Test")
    print("="*60)
    
    # Test 1: Load components
    print("\n1. Loading models...")
    try:
        predictor = DiseasePredictionModel(
            data_paths=['data/medicines.json'],
            scenario_path='data/scenarios.txt',
            use_scenarios_for_training=False,
        )
        preprocessor = TextPreprocessor()
        print("   ✓ All models loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading models: {e}")
        return False
    
    # Test 2: Load medicines.json
    print("\n2. Loading medicines.json...")
    try:
        with open('data/medicines.json', 'r', encoding='utf-8') as f:
            items = json.load(f)
        assert isinstance(items, list) and len(items) > 0
        print(f"   ✓ Loaded {len(items)} medicine items")
    except Exception as e:
        print(f"   ✗ Error loading medicines.json: {e}")
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
    
    # Test 4: Catalog symptom matching (basic)
    print("\n4. Testing medicines.json symptom matching...")
    try:
        from app_flask import _recommend_one_medicine_per_cluster
        symptoms = "fever headache body pain"
        recs = _recommend_one_medicine_per_cluster(symptoms, max_clusters=3)
        print(f"   Input: {symptoms}")
        print(f"   Detected groups: {[r.get('cluster_label') for r in recs]}")
        print(f"   Best medicines: {[(r.get('medicine') or {}).get('name') for r in recs]}")
        if not recs:
            print("   ✗ No recommendations returned")
            return False

        # Ensure we don't repeat the same medicine across groups.
        meds = [((r.get('medicine') or {}).get('name') or '').strip().lower() for r in recs]
        meds = [m for m in meds if m]
        if len(meds) != len(set(meds)):
            print("   ✗ Duplicate medicine recommendations found")
            return False

        # Ensure we provide a reason.
        if not (recs[0].get('why') or []):
            print("   ✗ Missing explanation for recommendation")
            return False

        print("   ✓ Matching works correctly")
    except Exception as e:
        print(f"   ✗ Error in matching: {e}")
        return False
    
    # Test 5: Scenario follow-ups are style-only (sanitized)
    print("\n5. Testing scenario style follow-ups...")
    try:
        from app_flask import _sanitize_style_line
        sample_lines = predictor.get_scenario_followups("I have fever and sore throat", top_k=8)
        cleaned = [s for s in (_sanitize_style_line(l) for l in sample_lines) if s]
        print(f"   Raw lines: {len(sample_lines)} | Clean lines: {len(cleaned)}")
        if not cleaned:
            print("   ✗ No style follow-up questions found (check scenarios.txt)")
            return False
        print(f"     Example: {cleaned[0]}")
        print("   ✓ Style follow-ups available")
    except Exception as e:
        print(f"   ✗ Error in style follow-ups: {e}")
        return False

    # Test 6: End-to-end simulation via Flask analyzer
    print("\n6. Running end-to-end simulation...")
    try:
        from app_flask import analyze_symptoms
        user_input = "I have fever, headache, and body pain"
        print(f"   User input: {user_input}")
        out = analyze_symptoms(user_input, user_name=None)
        ok = bool((out or {}).get('bot_message'))
        print(f"   Response generated: {ok}")
        if not ok:
            return False
        print("   ✓ End-to-end simulation successful")
    except Exception as e:
        print(f"   ✗ Error in end-to-end test: {e}")
        return False
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
    print("\nThe Medical AI Chatbot is ready to use!")
    print("Run: python app_flask.py")
    return True


if __name__ == "__main__":
    success = test_components()
    sys.exit(0 if success else 1)