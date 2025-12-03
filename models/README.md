# Model Files

This directory contains pre-trained model files for **BRIDGE** (BERT Representations for Identifying Depression via Gradient Estimators).

## Files

| File | Description | Size |
|------|-------------|------|
| `xgb.pkl` | XGBoost Classifier (Best performing ML model) | ~50MB |
| `logistic_regression.pkl` | Logistic Regression Classifier | ~10MB |
| `bernoulli_naive_bayes.pkl` | Bernoulli Naive Bayes Classifier | ~5MB |
| `multinomial_naive_bayes.pkl` | Multinomial Naive Bayes Classifier | ~5MB |
| `tfidf_vectorizer.pkl` | TF-IDF Vectorizer (50,000 features) | ~20MB |
| `label_encoder.pkl` | Label Encoder for class mapping | <1MB |
| `porter_stemmer.pkl` | Porter Stemmer for text preprocessing | <1MB |

## Usage

```python
import joblib

# Load model and preprocessing components
model = joblib.load('models/xgb.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
stemmer = joblib.load('models/porter_stemmer.pkl')
```

## Model Training

These models were trained using the notebooks in the `notebooks/` directory:
- `01_eda_bert_modeling.ipynb` - BERT-based approach
- `02_ml_modeling.ipynb` - Traditional ML approaches

## Note

⚠️ **Large Files**: Model files are large binary files. If you're having issues with Git, consider using [Git LFS](https://git-lfs.github.com/) for version control.
