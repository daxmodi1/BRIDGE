"""
BRIDGE Prediction Module
========================

This module provides functions to load trained models and make predictions
on mental health text data using the BRIDGE framework.
"""

import os
import joblib
import nltk
from scipy.sparse import hstack
from typing import Dict, Optional, Tuple, Any

from .preprocessing import clean_text, tokenize_text, stem_tokens


# Default model directory
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


class MentalHealthPredictor:
    """
    A class for predicting mental health status from text.
    
    Attributes:
        model: Trained classifier model
        vectorizer: TF-IDF vectorizer
        label_encoder: Label encoder for decoding predictions
        stemmer: Porter stemmer for text preprocessing
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the predictor with trained models.
        
        Args:
            model_dir (str, optional): Path to directory containing model files.
                                      Defaults to project's models/ directory.
        """
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.stemmer = None
        self._load_models()
    
    def _load_models(self):
        """Load all required model files."""
        try:
            self.model = joblib.load(os.path.join(self.model_dir, 'xgb.pkl'))
            self.vectorizer = joblib.load(os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'))
            self.label_encoder = joblib.load(os.path.join(self.model_dir, 'label_encoder.pkl'))
            self.stemmer = joblib.load(os.path.join(self.model_dir, 'porter_stemmer.pkl'))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model files not found in {self.model_dir}. "
                "Please ensure all model files are present."
            ) from e
    
    def predict(self, text: str) -> str:
        """
        Predict mental health status from text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            str: Predicted mental health category
            
        Example:
            >>> predictor = MentalHealthPredictor()
            >>> result = predictor.predict("I feel so hopeless")
            >>> print(result)
            'Depression'
        """
        # Preprocess text
        cleaned = clean_text(text)
        tokens = tokenize_text(cleaned)
        stemmed = stem_tokens(tokens, self.stemmer)
        
        # Extract features
        tfidf_features = self.vectorizer.transform([stemmed])
        num_features = [[len(text), len(nltk.sent_tokenize(text))]]
        combined = hstack([tfidf_features, num_features])
        
        # Predict
        prediction = self.model.predict(combined)
        return self.label_encoder.inverse_transform(prediction)[0]
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Get prediction probabilities for all classes.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict[str, float]: Dictionary mapping class names to probabilities
            
        Example:
            >>> predictor = MentalHealthPredictor()
            >>> probs = predictor.predict_proba("I feel anxious")
            >>> print(probs)
            {'Normal': 0.1, 'Anxiety': 0.6, ...}
        """
        # Preprocess text
        cleaned = clean_text(text)
        tokens = tokenize_text(cleaned)
        stemmed = stem_tokens(tokens, self.stemmer)
        
        # Extract features
        tfidf_features = self.vectorizer.transform([stemmed])
        num_features = [[len(text), len(nltk.sent_tokenize(text))]]
        combined = hstack([tfidf_features, num_features])
        
        # Get probabilities
        try:
            probabilities = self.model.predict_proba(combined)[0]
            classes = self.label_encoder.classes_
            return dict(zip(classes, probabilities))
        except AttributeError:
            # Model doesn't support predict_proba
            prediction = self.predict(text)
            return {prediction: 1.0}


def load_models(model_dir: Optional[str] = None) -> Tuple[Any, Any, Any, Any]:
    """
    Load all trained model components.
    
    Args:
        model_dir (str, optional): Path to model directory
        
    Returns:
        Tuple containing (model, vectorizer, label_encoder, stemmer)
    """
    model_dir = model_dir or DEFAULT_MODEL_DIR
    
    model = joblib.load(os.path.join(model_dir, 'xgb.pkl'))
    vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    stemmer = joblib.load(os.path.join(model_dir, 'porter_stemmer.pkl'))
    
    return model, vectorizer, label_encoder, stemmer


def predict_mental_health(text: str, model_dir: Optional[str] = None) -> str:
    """
    Convenience function to predict mental health status.
    
    Args:
        text (str): Input text to analyze
        model_dir (str, optional): Path to model directory
        
    Returns:
        str: Predicted mental health category
        
    Example:
        >>> result = predict_mental_health("I can't stop worrying about everything")
        >>> print(result)
        'Anxiety'
    """
    predictor = MentalHealthPredictor(model_dir)
    return predictor.predict(text)


if __name__ == "__main__":
    # Test the prediction functions
    test_texts = [
        "I feel so hopeless and can't see any way out of this darkness",
        "I'm having a great day! Everything is wonderful.",
        "I can't stop worrying about everything. My heart races constantly.",
        "I feel like I'm on top of the world one moment and in the depths of despair the next."
    ]
    
    print("Mental Health Prediction Demo")
    print("=" * 50)
    
    try:
        predictor = MentalHealthPredictor()
        
        for text in test_texts:
            prediction = predictor.predict(text)
            print(f"\nText: {text[:50]}...")
            print(f"Prediction: {prediction}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure model files are in the 'models/' directory.")
