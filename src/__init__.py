"""
BRIDGE: BERT Representations for Identifying Depression via Gradient Estimators
================================================================================

An AI-powered framework leveraging BERT embeddings and gradient boosting
to identify and classify mental health conditions from textual data.
"""

__version__ = "1.0.0"
__author__ = "BRIDGE Team"

from .preprocessing import preprocess_text, clean_text
from .predict import predict_mental_health, load_models

__all__ = [
    "preprocess_text",
    "clean_text", 
    "predict_mental_health",
    "load_models"
]
