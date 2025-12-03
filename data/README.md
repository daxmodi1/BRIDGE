# 📊 Dataset Information

## Overview

The **BRIDGE** project uses the **Sentiment Analysis for Mental Health** dataset from Kaggle.

## Dataset Source

📎 **Download Link**: [Sentiment Analysis for Mental Health - Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

## Data Sources

The dataset is a combination of multiple mental health related datasets:

| Source Dataset | Description |
|----------------|-------------|
| 3k Conversations Dataset for Chatbot | Conversational data |
| Depression Reddit Cleaned | Reddit posts related to depression |
| Human Stress Prediction | Stress-related text data |
| Predicting Anxiety in Mental Health Data | Anxiety-related statements |
| Mental Health Dataset Bipolar | Bipolar disorder data |
| Reddit Mental Health Data | General mental health discussions |
| Students Anxiety and Depression Dataset | Student mental health data |
| Suicidal Mental Health Dataset | Suicidal ideation data |
| Suicidal Tweet Detection Dataset | Twitter data on suicidal content |

## Dataset Structure

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `statement` | string | Text statement from an individual |
| `status` | string | Mental health category label |

### Target Classes

| Class | Description |
|-------|-------------|
| Normal | No mental health concerns |
| Depression | Depressive symptoms |
| Suicidal | Suicidal ideation |
| Anxiety | Anxiety symptoms |
| Stress | Stress-related |
| Bi-Polar | Bipolar disorder |
| Personality Disorder | Personality disorder symptoms |

## Usage Instructions

1. Download `Combined Data.csv` from the Kaggle link above
2. Place the file in this `data/` directory
3. The notebooks will automatically load the data from here

## Data Privacy Notice

⚠️ **Important**: This dataset contains sensitive mental health-related text. Please:
- Use this data responsibly for research and educational purposes only
- Do not attempt to identify individuals from the data
- Follow ethical guidelines when working with mental health data

## Citation

If you use this dataset, please cite the original Kaggle dataset:

```
@dataset{mental_health_sentiment,
  author = {Suchintika Sarkar},
  title = {Sentiment Analysis for Mental Health},
  year = {2024},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health}
}
```
