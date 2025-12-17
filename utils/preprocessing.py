"""
NLP Preprocessing Utilities for Medical Chatbot
Handles text preprocessing and symptom extraction
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK datasets"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

class TextPreprocessor:
    """Preprocesses text for medical symptom analysis"""
    
    def __init__(self):
        download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Keep medical-relevant words
        self.medical_stopwords = self.stop_words - {
            'pain', 'fever', 'no', 'not', 'severe', 'mild', 'high', 'low'
        }
    
    def clean_text(self, text):
        """
        Clean and normalize input text
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords while keeping medical terms
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Filtered tokens
        """
        return [token for token in tokens if token not in self.medical_stopwords]
    
    def lemmatize(self, tokens):
        """
        Lemmatize tokens to base form
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Raw input text
            
        Returns:
            list: Preprocessed tokens
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove stopwords
        filtered_tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        lemmatized_tokens = self.lemmatize(filtered_tokens)
        
        return lemmatized_tokens
    
    def extract_symptoms(self, text):
        """
        Extract and normalize symptoms from text
        
        Args:
            text (str): Raw symptom description
            
        Returns:
            list: List of extracted symptoms
        """
        # Preprocess text
        tokens = self.preprocess(text)
        
        # Join tokens back
        normalized_text = ' '.join(tokens)
        
        return normalized_text


def create_symptom_features(symptoms_list, all_symptoms):
    """
    Create feature vector from symptoms
    
    Args:
        symptoms_list (list): List of user symptoms
        all_symptoms (list): List of all possible symptoms
        
    Returns:
        list: Binary feature vector
    """
    feature_vector = []
    for symptom in all_symptoms:
        if symptom in symptoms_list:
            feature_vector.append(1)
        else:
            feature_vector.append(0)
    return feature_vector