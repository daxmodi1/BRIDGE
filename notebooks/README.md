# Jupyter Notebooks

This directory contains the main analysis notebooks for the **BRIDGE** project (BERT Representations for Identifying Depression via Gradient Estimators).

## Notebooks

### 1. `01_eda_bert_modeling.ipynb`

**Focus**: Exploratory Data Analysis + BERT-based Deep Learning

**Contents**:
- 📊 Comprehensive EDA with visualizations
- 📝 Text feature engineering
- 🔤 Word cloud generation for each mental health category
- 🧠 BERT embeddings extraction
- 🤖 XGBoost classification with BERT features
- 📈 **94% accuracy achieved**

**Key Sections**:
1. Importing Libraries and Reading Data
2. EDA and Data Preparation
3. Text Analysis and Visualization
4. BERT Embeddings Generation
5. Model Training and Evaluation
6. Results and Confusion Matrix

---

### 2. `02_ml_modeling.ipynb`

**Focus**: Traditional Machine Learning Approaches

**Contents**:
- 📊 Dataset overview and statistics
- 📝 Text preprocessing (tokenization, stemming)
- 🔢 TF-IDF feature extraction
- ⚖️ Class balancing with resampling
- 🤖 Multiple ML model comparison
- 💾 Model saving and export

**Models Implemented**:
- Logistic Regression
- Bernoulli Naive Bayes
- Multinomial Naive Bayes
- XGBoost

**Key Sections**:
1. Exploratory Data Analysis
2. Target Variable Distribution
3. Text Data Analysis
4. Preprocessing and Feature Engineering
5. Model Training and Comparison
6. Model Export

---

## Running the Notebooks

### Prerequisites

1. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) and place it in the `data/` folder.

### Execution

```bash
# Start Jupyter
cd notebooks
jupyter notebook

# Or use JupyterLab
jupyter lab
```

### Execution Order

For best results, run the notebooks in order:
1. `01_eda_bert_modeling.ipynb` - For understanding data and BERT approach
2. `02_ml_modeling.ipynb` - For traditional ML models and saved model generation

---

## Hardware Requirements

| Notebook | CPU | GPU | RAM | Time |
|----------|-----|-----|-----|------|
| 01_eda_bert_modeling | ✓ | Recommended | 16GB+ | ~2 hours |
| 02_ml_modeling | ✓ | Optional | 8GB+ | ~30 min |

---

## Output

Both notebooks generate:
- 📊 Visualizations (distribution plots, word clouds, confusion matrices)
- 📈 Model metrics (accuracy, precision, recall, F1-score)
- 💾 Saved models (in `models/` directory)
