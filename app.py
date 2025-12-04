"""
BRIDGE - Mental Health Text Classification App
===============================================

A Streamlit web application for classifying mental health conditions
from text using the BRIDGE framework with TF-IDF + XGBoost.

Run with: streamlit run app.py
"""

import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack
import os

import warnings
warnings.filterwarnings('ignore')

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="BRIDGE - Mental Health Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk_data()

# Load models
@st.cache_resource
def load_models():
    """Load all required model files."""
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    model = joblib.load(os.path.join(model_dir, 'xgb.pkl'))
    vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    stemmer = joblib.load(os.path.join(model_dir, 'porter_stemmer.pkl'))
    
    return model, vectorizer, label_encoder, stemmer

# Text preprocessing
def preprocess_text(text, stemmer):
    """Clean and preprocess text for prediction."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove markdown-style links
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    
    text = re.sub(r'@\w+', '', text)
    
    text = re.sub(r'[^\w\s]', '', text)
    
    tokens = word_tokenize(text)
    stemmed = ' '.join(stemmer.stem(token) for token in tokens)
    
    return text, stemmed

# Prediction function
def predict_mental_health(text, model, vectorizer, label_encoder, stemmer):
    """Predict mental health status from text."""
    cleaned_text, stemmed_text = preprocess_text(text, stemmer)
    
    # Extract features
    tfidf_features = vectorizer.transform([stemmed_text])
    num_features = [[len(cleaned_text), len(nltk.sent_tokenize(text))]]
    combined = hstack([tfidf_features, num_features])
    
    # Predict
    prediction = model.predict(combined)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    
    # Get probabilities if available
    try:
        probabilities = model.predict_proba(combined)[0]
        prob_dict = dict(zip(label_encoder.classes_, probabilities))
    except:
        prob_dict = {predicted_label: 1.0}
    
    return predicted_label, prob_dict

# Category info
CATEGORY_INFO = {
    "Normal": {
        "emoji": "🟢",
        "color": "#28a745",
        "description": "No significant mental health concerns detected."
    },
    "Depression": {
        "emoji": "🔵",
        "color": "#007bff",
        "description": "Signs of depressive symptoms detected."
    },
    "Suicidal": {
        "emoji": "🔴",
        "color": "#dc3545",
        "description": "Indicators of suicidal ideation detected. Please seek help immediately."
    },
    "Anxiety": {
        "emoji": "🟡",
        "color": "#ffc107",
        "description": "Anxiety-related expressions detected."
    },
    "Stress": {
        "emoji": "🟠",
        "color": "#fd7e14",
        "description": "Stress-related patterns detected."
    },
    "Bipolar": {
        "emoji": "🟣",
        "color": "#6f42c1",
        "description": "Bipolar disorder indicators detected."
    },
    "Personality disorder": {
        "emoji": "⚪",
        "color": "#6c757d",
        "description": "Personality disorder patterns detected."
    }
}

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>BRIDGE</h1>
    <p><b>B</b>ERT <b>R</b>epresentations for <b>I</b>dentifying <b>D</b>epression via <b>G</b>radient <b>E</b>stimators</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### About BRIDGE")
    st.markdown("""
    BRIDGE is an AI-powered mental health classification system using 
    **TF-IDF** and **XGBoost**.
    
    **Supported Categories:**
    - 🟢 Normal
    - 🔵 Depression
    - 🔴 Suicidal
    - 🟡 Anxiety
    - 🟠 Stress
    - 🟣 Bipolar
    - ⚪ Personality Disorder
    """)

# Initialize session state for example text
if 'example_text' not in st.session_state:
    st.session_state.example_text = ""

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter Text for Analysis")
    
    # Text input
    user_text = st.text_area(
        "Type or paste the text you want to analyze:",
        value=st.session_state.example_text,
        height=200,
        placeholder="Enter the text here... For example: 'I've been feeling really down lately and can't seem to find joy in things I used to love.'"
    )
    
    # Example texts
    st.markdown("**Or try an example:**")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("Normal", use_container_width=True):
            st.session_state.example_text = "I had a great day today! Went for a walk in the park and enjoyed the sunshine. Life is good."
            st.rerun()
    
    with example_col2:
        if st.button("Depression", use_container_width=True):
            st.session_state.example_text = "I feel so empty inside. Nothing brings me joy anymore and I just want to stay in bed all day."
            st.rerun()
    
    with example_col3:
        if st.button("Anxiety", use_container_width=True):
            st.session_state.example_text = "My heart is racing and I can't stop worrying about everything. What if something bad happens?"
            st.rerun()
    
    # Analyze button
    analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📊 Classification Results")
    
    if analyze_button and user_text.strip():
        try:
            with st.spinner("Loading model..."):
                model, vectorizer, label_encoder, stemmer = load_models()
            
            with st.spinner("Analyzing..."):
                prediction, probabilities = predict_mental_health(
                    user_text, model, vectorizer, label_encoder, stemmer
                )
            
            # Get category info
            category = CATEGORY_INFO.get(prediction, {"emoji": "❓", "color": "#999", "description": "Classification result."})
            
            # Display result
            st.markdown(f"""
            <div class="result-box" style="background: {category['color']}20; border: 2px solid {category['color']};">
                <h1 style="color: {category['color']}; font-size: 3rem;">{category['emoji']}</h1>
                <h2 style="color: {category['color']}; margin: 0.5rem 0;">{prediction}</h2>
                <p style="color: #666;">{category['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability distribution
            st.markdown("#### Confidence Scores")
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            
            for label, prob in sorted_probs:
                cat_info = CATEGORY_INFO.get(label, {"emoji": "❓", "color": "#999"})
                prob_float = float(prob)
                st.progress(prob_float, text=f"{cat_info['emoji']} {label}: {prob_float*100:.1f}%")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure all model files are in the 'models/' directory.")
    
    elif analyze_button:
        st.warning("Please enter some text to analyze.")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #999;">
            <p>Enter text and click <b>Analyze</b> to see results</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🌉 <b>BRIDGE</b> - Building connections through AI for Mental Health Awareness</p>
    <p style="font-size: 0.8rem;">Made with ❤️ using Streamlit | <a href="https://github.com/daxmodi1/BRIDGE">GitHub</a></p>
</div>
""", unsafe_allow_html=True)
