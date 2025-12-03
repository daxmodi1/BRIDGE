# 📊 BRIDGE Model Comparison

This document provides a detailed comparison of all machine learning models used in the BRIDGE framework.

## Overview

We implemented and compared multiple machine learning approaches:

1. **BERT + XGBoost** (Deep Learning)
2. **XGBoost with TF-IDF** (Gradient Boosting)
3. **Logistic Regression** (Linear Model)
4. **Bernoulli Naive Bayes** (Probabilistic)
5. **Multinomial Naive Bayes** (Probabilistic)

---

## Model Details

### 1. BERT + XGBoost (94% Accuracy) ⭐

**Architecture:**
- BERT encoder: `bert_en_uncased_L-12_H-768_A-12`
- Classifier: XGBoost with optimized hyperparameters

**Hyperparameters:**
```python
XGBClassifier(
    alpha=0.5,
    lambda_=1.0,
    learning_rate=0.05,
    n_estimators=500,
    early_stopping_rounds=10
)
```

**Strengths:**
- Captures contextual meaning of text
- State-of-the-art NLP representations
- Handles nuanced language patterns

**Limitations:**
- Computationally expensive
- Requires GPU for efficient training
- Large model size

---

### 2. XGBoost with TF-IDF (~85% Accuracy)

**Features:**
- TF-IDF vectors with n-grams (1, 2)
- Maximum 50,000 features
- Additional numerical features (character count, sentence count)

**Hyperparameters:**
```python
XGBClassifier(
    learning_rate=0.2,
    max_depth=7,
    n_estimators=500,
    random_state=101
)
```

**Strengths:**
- Fast inference time
- Handles imbalanced data well
- Feature importance interpretability

---

### 3. Logistic Regression (~78% Accuracy)

**Configuration:**
```python
LogisticRegression(
    solver='liblinear',
    penalty='l1',
    C=10,
    random_state=101
)
```

**Strengths:**
- Simple and interpretable
- Fast training and inference
- Good baseline model

---

### 4. Bernoulli Naive Bayes (~72% Accuracy)

**Configuration:**
```python
BernoulliNB(
    alpha=0.1,
    binarize=0.0
)
```

**Strengths:**
- Works well with binary features
- Fast and efficient
- Good for text classification

---

### 5. Multinomial Naive Bayes (~70% Accuracy)

**Configuration:**
```python
MultinomialNB(
    alpha=0.1
)
```

**Strengths:**
- Designed for text classification
- Handles word frequencies well
- Extremely fast

---

## Performance Comparison

| Model | Accuracy | Training Time | Inference Time | Model Size |
|-------|----------|---------------|----------------|------------|
| BERT + XGBoost | **94%** | ~2 hours | ~100ms | ~400MB |
| XGBoost (TF-IDF) | 85% | ~10 min | ~5ms | ~50MB |
| Logistic Regression | 78% | ~2 min | ~1ms | ~10MB |
| Bernoulli NB | 72% | ~30 sec | <1ms | ~5MB |
| Multinomial NB | 70% | ~30 sec | <1ms | ~5MB |

---

## Class-wise Performance (Best Model: BERT + XGBoost)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 0.95 | 0.94 | 0.94 |
| Depression | 0.93 | 0.94 | 0.93 |
| Suicidal | 0.95 | 0.96 | 0.95 |
| Anxiety | 0.92 | 0.91 | 0.91 |
| Stress | 0.93 | 0.92 | 0.92 |
| Bi-Polar | 0.94 | 0.95 | 0.94 |
| Personality Disorder | 0.92 | 0.93 | 0.92 |

---

## Recommendations

### For Production Use:
- **High Accuracy Required**: Use BERT + XGBoost
- **Fast Inference Required**: Use XGBoost with TF-IDF
- **Resource Constrained**: Use Logistic Regression

### For Research:
- Explore other transformer models (RoBERTa, DistilBERT)
- Try ensemble methods combining multiple models
- Investigate attention mechanisms for interpretability

---

## Hyperparameter Tuning

All models were tuned using GridSearchCV with 5-fold cross-validation. The search spaces and optimal parameters are documented in the notebooks.
