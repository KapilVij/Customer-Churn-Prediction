# Customer Churn Prediction with Machine Learning

This project uses a machine learning pipeline to predict **customer churn** (whether a customer will leave or stay) based on their attributes and service usage data. It includes data visualizing, preprocessing, feature engineering, handling class imbalance, model training using a Random Forest Classifier, and evaluation.

---

## Project Overview

Customer churn is a major concern for subscription-based businesses. Accurately predicting which customers are likely to leave helps companies take proactive measures to retain them.

In this project, we:
- Understand all the variables (Using visualizations)
- Preprocess the dataset (label encoding, one-hot encoding)
- Balance the dataset using oversampling
- Train a Random Forest classifier
- Evaluate model performance using accuracy, precision, recall, and F1-score

---

## Dataset
The dataset contains information about telecom customers, including:
- Demographic data (e.g. gender, senior citizen)
- Service-related data (e.g. phone, internet service)
- Account information (e.g. contract type, payment method)
- **Target variable**: `Churn` (Yes/No)

---

## Libraries Used

```python
pandas
numpy
matplotlib
seaborn
sklearn
imblearn
```

---

## Steps Followed in ML

### 1. Data Preprocessing
- Target variable `Churn` converted to binary (Yes → 1, No → 0)
- Label encoding for binary columns (e.g. gender, Partner)
- One-hot encoding for multi-class categorical columns

### 2. Handling Imbalanced Data
- Used **RandomOverSampler** to oversample the minority class (churned customers) for a balanced dataset.

### 3. Train-Test Split
- Ensuring 80-20 splitting.

### 4. Model Training
- RandomForestClassifier with 100 trees (`n_estimators=100`)

### 5. Model Evaluation
- Metrics used: **Confusion Matrix**, **Classification Report**, **Accuracy**
- Output:

```
Confusion Matrix:
 [[856 177]
 [ 49 988]]

Classification Report:
               precision    recall  f1-score   support

           0       0.95      0.83      0.88      1033
           1       0.85      0.95      0.90      1037

    accuracy                           0.89      2070
   macro avg       0.90      0.89      0.89      2070
weighted avg       0.90      0.89      0.89      2070


Accuracy: 0.8908212560386474
```

---

## Model Evaluation Interpretation

- **Accuracy** of the model is **89%**, indicating strong overall performance.
- **Class 0 (No Churn)**:
  - **Precision**: 94% — When the model predicts a customer will *not churn*, it's right 94% of the time.
  - **Recall**: 83% — The model correctly identifies 83% of customers who actually did *not churn*.
- **Class 1 (Churn)**:
  - **Precision**: 85% — When the model predicts a customer will *churn*, it's correct 85% of the time.
  - **Recall**: 95% — The model correctly identifies 95% of customers who actually *churned*.

---
