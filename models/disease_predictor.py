"""
Disease Prediction Model for Medical Chatbot
Uses machine learning to predict diseases based on symptoms
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os


class DiseasePredictionModel:
    """Machine learning model for disease prediction"""

    def __init__(self, data_paths=None):
        """
        Initialize the disease prediction model

        Args:
            data_paths (str or list): Path or list of paths to medicines/diseases JSON files
        """
        if data_paths is None:
            data_paths = ['data/medicines.json']
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        self.data_paths = data_paths
        self.diseases = []
        self.symptoms_map = {}
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.model = MultinomialNB()
        self.is_trained = False
        # Load disease data from one or more JSON files
        self.load_data()
    
    def load_data(self):
        """Load disease and symptom data from one or more JSON files and merge duplicates."""
        merged = {}

        for path in self.data_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    items = data.get('diseases', []) if isinstance(data, dict) else data

                    for disease in items:
                        # Normalize name
                        name = disease.get('name', '').strip()
                        if not name:
                            continue

                        if name in merged:
                            # Merge symptoms
                            existing = merged[name]
                            existing_symptoms = set(existing.get('symptoms', []))
                            for s in disease.get('symptoms', []):
                                if s not in existing_symptoms:
                                    existing.setdefault('symptoms', []).append(s)
                                    existing_symptoms.add(s)

                            # Merge medicines by name
                            existing_meds = existing.setdefault('medicines', [])
                            existing_med_names = set(m.get('name') for m in existing_meds)
                            for m in disease.get('medicines', []):
                                if m.get('name') not in existing_med_names:
                                    existing_meds.append(m)
                                    existing_med_names.add(m.get('name'))
                        else:
                            # Add a shallow copy to avoid modifying source
                            merged[name] = {
                                'name': name,
                                'symptoms': list(disease.get('symptoms', [])),
                                'medicines': list(disease.get('medicines', []))
                            }
            except FileNotFoundError:
                print(f"Warning: Could not find data file at {path}")
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON file at {path}")

        # Convert merged dict to list
        self.diseases = list(merged.values())

        # Build symptom -> diseases mapping
        self.symptoms_map = {}
        for disease in self.diseases:
            disease_name = disease['name']
            for symptom in disease.get('symptoms', []):
                if symptom not in self.symptoms_map:
                    self.symptoms_map[symptom] = []
                self.symptoms_map[symptom].append(disease_name)
    
    def prepare_training_data(self):
        """
        Prepare training data from disease information
        
        Returns:
            tuple: (X_train, y_train) feature vectors and labels
        """
        X = []  # Symptom descriptions
        y = []  # Disease labels
        
        for disease in self.diseases:
            disease_name = disease['name']
            symptoms = disease['symptoms']
            
            # Create training examples
            # Each symptom combination creates a training instance
            symptom_text = ' '.join(symptoms)
            X.append(symptom_text)
            y.append(disease_name)
            
            # Add variations with different symptom combinations
            if len(symptoms) > 2:
                for i in range(len(symptoms)):
                    # Create partial symptom combinations by removing one symptom
                    partial_symptoms = symptoms[:i] + symptoms[i+1:]
                    if len(partial_symptoms) >= 2:
                        symptom_text = ' '.join(partial_symptoms)
                        X.append(symptom_text)
                        y.append(disease_name)
        
        return X, y
    
    def train(self):
        """Train the disease prediction model"""
        if not self.diseases:
            print("No disease data available for training")
            return False
        
        # Prepare training data
        X, y = self.prepare_training_data()
        
        if len(X) == 0:
            print("No training data available")
            return False
        
        # Transform text to TF-IDF features
        X_vectorized = self.vectorizer.fit_transform(X)
        
        # Train the model
        self.model.fit(X_vectorized, y)
        self.is_trained = True
        
        print(f"Model trained with {len(self.diseases)} diseases")
        return True
    
    def predict(self, symptoms_text):
        """
        Predict disease based on symptoms
        
        Args:
            symptoms_text (str): User's symptom description
            
        Returns:
            list: List of tuples (disease_name, probability)
        """
        if not self.is_trained:
            self.train()
        
        # Transform input symptoms
        X = self.vectorizer.transform([symptoms_text])
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)[0]
        diseases = self.model.classes_
        
        # Create list of (disease, probability) tuples
        predictions = list(zip(diseases, probabilities))
        
        # Sort by probability (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top predictions with probability > 0.01
        return [(disease, prob) for disease, prob in predictions if prob > 0.01]
    
    def get_disease_info(self, disease_name):
        """
        Get detailed information about a disease
        
        Args:
            disease_name (str): Name of the disease
            
        Returns:
            dict: Disease information including symptoms and medicines
        """
        for disease in self.diseases:
            if disease['name'].lower() == disease_name.lower():
                return disease
        return None
    
    def match_symptoms(self, user_symptoms):
        """
        Match user symptoms with diseases using rule-based approach
        
        Args:
            user_symptoms (list): List of user symptom keywords
            
        Returns:
            list: List of matching diseases with match scores
        """
        disease_scores = {}
        
        for disease in self.diseases:
            disease_name = disease['name']
            disease_symptoms = [s.lower() for s in disease['symptoms']]
            
            # Calculate match score
            matches = 0
            for user_symptom in user_symptoms:
                user_symptom = user_symptom.lower()
                # Check for exact or partial matches
                for disease_symptom in disease_symptoms:
                    if user_symptom in disease_symptom or disease_symptom in user_symptom:
                        matches += 1
                        break
            
            if matches > 0:
                # Calculate match percentage
                match_percentage = (matches / len(disease_symptoms)) * 100
                disease_scores[disease_name] = {
                    'matches': matches,
                    'total_symptoms': len(disease_symptoms),
                    'match_percentage': match_percentage
                }
        
        # Sort by number of matches
        sorted_diseases = sorted(
            disease_scores.items(),
            key=lambda x: (x[1]['matches'], x[1]['match_percentage']),
            reverse=True
        )
        
        return sorted_diseases
    
    def save_model(self, model_dir='models'):
        """Save trained model and vectorizer"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        if self.is_trained:
            joblib.dump(self.model, os.path.join(model_dir, 'disease_model.pkl'))
            joblib.dump(self.vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
            print("Model saved successfully")
        else:
            print("Model not trained yet")
    
    def load_model(self, model_dir='models'):
        """Load pre-trained model and vectorizer"""
        try:
            self.model = joblib.load(os.path.join(model_dir, 'disease_model.pkl'))
            self.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))
            self.is_trained = True
            print("Model loaded successfully")
            return True
        except FileNotFoundError:
            print("No saved model found. Training new model...")
            return False