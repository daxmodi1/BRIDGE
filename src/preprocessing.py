"""
BRIDGE Text Preprocessing Module
================================

This module contains functions for preprocessing text data
for the BRIDGE mental health classification framework.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from typing import List, Optional


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing URLs, mentions, and special characters.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
        
    Example:
        >>> clean_text("Check this https://example.com @user!")
        'Check this'
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove markdown-style links
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    
    # Remove handles/mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into individual words.
    
    Args:
        text (str): Input text to tokenize
        
    Returns:
        List[str]: List of tokens
        
    Example:
        >>> tokenize_text("I feel happy today")
        ['I', 'feel', 'happy', 'today']
    """
    if not text:
        return []
    
    return word_tokenize(text)


def stem_tokens(tokens: List[str], stemmer: Optional[PorterStemmer] = None) -> str:
    """
    Apply stemming to a list of tokens and join them.
    
    Args:
        tokens (List[str]): List of word tokens
        stemmer (PorterStemmer, optional): Stemmer instance. Creates new if None.
        
    Returns:
        str: Space-separated stemmed tokens
        
    Example:
        >>> stem_tokens(['running', 'happily'])
        'run happili'
    """
    if stemmer is None:
        stemmer = PorterStemmer()
    
    return ' '.join(stemmer.stem(str(token)) for token in tokens)


def preprocess_text(text: str, stemmer: Optional[PorterStemmer] = None) -> str:
    """
    Full preprocessing pipeline: clean, tokenize, and stem text.
    
    Args:
        text (str): Input text to preprocess
        stemmer (PorterStemmer, optional): Stemmer instance
        
    Returns:
        str: Preprocessed text ready for vectorization
        
    Example:
        >>> preprocess_text("I'm feeling very anxious today!")
        'im feel veri anxiou today'
    """
    # Clean text
    cleaned = clean_text(text)
    
    # Tokenize
    tokens = tokenize_text(cleaned)
    
    # Stem and join
    stemmed = stem_tokens(tokens, stemmer)
    
    return stemmed


def extract_features(text: str) -> dict:
    """
    Extract numerical features from text.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary containing extracted features
        
    Example:
        >>> features = extract_features("Hello world!")
        >>> print(features['num_characters'])
        12
    """
    clean = clean_text(text)
    
    return {
        'num_characters': len(text),
        'num_words': len(text.split()),
        'num_sentences': len(nltk.sent_tokenize(text)),
        'avg_word_length': sum(len(word) for word in text.split()) / max(len(text.split()), 1),
        'vocabulary_size': len(set(text.split()))
    }


if __name__ == "__main__":
    # Test the preprocessing functions
    sample_text = "I've been feeling really anxious lately... Can't sleep at night. https://example.com @someone"
    
    print("Original text:")
    print(sample_text)
    print("\nCleaned text:")
    print(clean_text(sample_text))
    print("\nPreprocessed text:")
    print(preprocess_text(sample_text))
    print("\nExtracted features:")
    print(extract_features(sample_text))
