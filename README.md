<div align="center">

# 🌉 BRIDGE

### **B**ERT **R**epresentations for **I**dentifying **D**epression via **G**radient **E**stimators

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/BERT%20Accuracy-94%25-brightgreen.svg)]()

<p align="center">
  <img src="docs/images/bridge_banner.png" alt="BRIDGE - Mental Health Analysis" width="600">
</p>

*An AI-powered framework leveraging BERT embeddings and gradient boosting to identify and classify mental health conditions from textual data*

[📊 View Notebooks](#-notebooks) • [🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🤝 Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Notebooks](#-notebooks)
- [Models](#-models)
- [Dataset](#-dataset)
- [Results](#-results)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 Overview

**BRIDGE** (BERT Representations for Identifying Depression via Gradient Estimators) is an advanced AI framework designed to analyze and classify mental health conditions from textual data. Mental health is a critical aspect of overall well-being, and understanding the nuances of mental health conditions can be a powerful tool in providing timely support and interventions.

By leveraging **BERT embeddings** combined with **Gradient Boosting (XGBoost)**, BRIDGE achieves **94% accuracy** in classifying statements into **7 mental health categories**:

| Category | Description |
|----------|-------------|
| 🟢 **Normal** | No significant mental health concerns |
| 🔵 **Depression** | Signs of depressive symptoms |
| 🔴 **Suicidal** | Indicators of suicidal ideation |
| 🟡 **Anxiety** | Anxiety-related expressions |
| 🟠 **Stress** | Stress-related patterns |
| 🟣 **Bi-Polar** | Bipolar disorder indicators |
| ⚪ **Personality Disorder** | Personality disorder patterns |

---

## ✨ Key Features

- **🔬 Comprehensive EDA**: In-depth exploratory data analysis with visualizations
- **📝 Text Preprocessing**: Advanced NLP pipeline with tokenization, stemming, and cleaning
- **🤖 Multiple Models**: Implementation of various ML algorithms for comparison
- **🧠 BERT Integration**: State-of-the-art transformer-based embeddings achieving **94% accuracy**
- **📊 TF-IDF Features**: Traditional NLP feature extraction with n-grams
- **⚖️ Class Balancing**: SMOTE and resampling techniques for handling imbalanced data
- **💾 Model Persistence**: Pre-trained models saved for inference
- **📈 Detailed Metrics**: Confusion matrices, classification reports, and accuracy scores

---

## 📁 Project Structure

```
BRIDGE/
│
├── 📓 notebooks/                    # Jupyter notebooks
│   ├── 01_eda_bert_modeling.ipynb   # EDA + BERT model (94% accuracy)
│   └── 02_ml_modeling.ipynb         # Traditional ML models
│
├── 🤖 models/                       # Saved trained models
│   ├── bernoulli_naive_bayes.pkl
│   ├── multinomial_naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── xgb.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   └── porter_stemmer.pkl
│
├── 📊 data/                         # Dataset directory
│   └── README.md                    # Data source information
│
├── 📂 src/                          # Source code
│   ├── __init__.py
│   ├── preprocessing.py             # Text preprocessing utilities
│   └── predict.py                   # Prediction functions
│
├── 📚 docs/                         # Documentation
│   ├── images/                      # Images for documentation
│   └── model_comparison.md          # Detailed model comparison
│
├── 📄 README.md                     # Project documentation
├── 📋 requirements.txt              # Python dependencies
├── 📜 LICENSE                       # MIT License
├── 🔧 .gitignore                    # Git ignore rules
└── 🤝 CONTRIBUTING.md               # Contribution guidelines
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/BRIDGE.git
   cd BRIDGE
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Linux/MacOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Get the dataset from [Kaggle: Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
   - Place `Combined Data.csv` in the `data/` directory

---

## 🚀 Quick Start

### Using Pre-trained Models

```python
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack

# Load models
model = joblib.load('models/xgb.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
stemmer = joblib.load('models/porter_stemmer.pkl')

def predict_mental_health(text):
    """
    Predict mental health status from text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        str: Predicted mental health category
    """
    # Preprocess
    text = text.lower()
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize and stem
    tokens = word_tokenize(text)
    stemmed = ' '.join(stemmer.stem(token) for token in tokens)
    
    # Extract features
    tfidf_features = vectorizer.transform([stemmed])
    num_features = [[len(text), len(nltk.sent_tokenize(text))]]
    combined = hstack([tfidf_features, num_features])
    
    # Predict
    prediction = model.predict(combined)
    return label_encoder.inverse_transform(prediction)[0]

# Example usage
result = predict_mental_health("I feel so hopeless and can't see any way out")
print(f"Predicted Status: {result}")
```

---

## 📓 Notebooks

| Notebook | Description | Key Highlights |
|----------|-------------|----------------|
| **01_eda_bert_modeling.ipynb** | EDA + BERT-based classification | BERT embeddings, XGBoost, **94% accuracy** |
| **02_ml_modeling.ipynb** | Traditional ML approaches | TF-IDF, Multiple classifiers, Feature engineering |

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

---

## 🤖 Models

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **BERT + XGBoost** | **94%** | 0.94 | 0.94 | 0.94 |
| XGBoost (TF-IDF) | ~85% | 0.85 | 0.84 | 0.84 |
| Logistic Regression | ~78% | 0.78 | 0.77 | 0.77 |
| Bernoulli Naive Bayes | ~72% | 0.72 | 0.71 | 0.71 |
| Multinomial Naive Bayes | ~70% | 0.70 | 0.69 | 0.69 |

### Saved Models

All trained models are saved in the `models/` directory:

- `xgb.pkl` - XGBoost Classifier
- `logistic_regression.pkl` - Logistic Regression
- `bernoulli_naive_bayes.pkl` - Bernoulli Naive Bayes
- `multinomial_naive_bayes.pkl` - Multinomial Naive Bayes
- `tfidf_vectorizer.pkl` - TF-IDF Vectorizer
- `label_encoder.pkl` - Label Encoder
- `porter_stemmer.pkl` - Porter Stemmer

---

## 📊 Dataset

### Source

The dataset integrates information from multiple Kaggle datasets:

- 3k Conversations Dataset for Chatbot
- Depression Reddit Cleaned
- Human Stress Prediction
- Predicting Anxiety in Mental Health Data
- Mental Health Dataset Bipolar
- Reddit Mental Health Data
- Students Anxiety and Depression Dataset
- Suicidal Mental Health Dataset
- Suicidal Tweet Detection Dataset

📎 **Download**: [Sentiment Analysis for Mental Health - Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

### Dataset Statistics

| Attribute | Value |
|-----------|-------|
| Total Samples | ~52,000+ |
| Features | 2 (statement, status) |
| Classes | 7 |
| Class Distribution | Imbalanced (addressed via resampling) |

---

## 📈 Results

### Key Findings from EDA

1. **Text Length Analysis**: Normal individuals tend to write shorter statements compared to those with mental health conditions
2. **Word Cloud Insights**: Distinct vocabulary patterns emerge for different mental health categories
3. **Class Distribution**: Dataset shows imbalance with "Normal" being the majority class
4. **Correlation**: Statement length and vocabulary size correlate with certain conditions

### Model Performance

The **BERT + XGBoost** combination achieved the highest accuracy of **94%**, demonstrating the power of transformer-based embeddings for mental health text classification.

---

## 📖 Documentation

Detailed documentation is available in the `docs/` directory:

- [Model Comparison](docs/model_comparison.md) - Detailed analysis of all models
- [Data Dictionary](docs/data_dictionary.md) - Feature descriptions
- [Methodology](docs/methodology.md) - Approach and techniques used

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Kaggle](https://www.kaggle.com/) for hosting the dataset
- [TensorFlow Hub](https://tfhub.dev/) for BERT models
- [scikit-learn](https://scikit-learn.org/) for ML algorithms
- [XGBoost](https://xgboost.readthedocs.io/) for gradient boosting
- All contributors to the original datasets

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

**BRIDGE** - *Building connections through AI for Mental Health Awareness* 🌉💚

</div>
#   B R I D G E  
 