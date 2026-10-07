# Credit Card Default Prediction

## 📌 Project Overview

Credit Card Default Prediction is an end-to-end machine learning classification project that predicts whether a credit card customer is likely to default on their payment.

The project uses customer demographic information, credit limit, historical repayment status, bill amounts, and payment amounts to identify customers who may be at higher risk of default.

The project follows a complete machine learning workflow, from exploratory data analysis and feature engineering to model comparison, cross-validation, hyperparameter tuning, evaluation, threshold analysis, model interpretation, and deployment preparation.

---

## 🎯 Business Problem

Credit card default is a significant risk for financial institutions.

The objective of this project is to build a machine learning model that can identify customers who are more likely to default.

A reliable prediction system can help financial institutions:

- Identify potentially high-risk customers
- Support credit-risk assessment
- Prioritize customers for early intervention
- Reduce potential financial losses
- Improve data-driven decision-making

In credit-risk prediction, incorrectly classifying an actual defaulter as a non-defaulter can be costly.

Therefore, this project does not rely on accuracy alone and evaluates the model using multiple classification metrics.

---

## 🎯 Project Objectives

- Perform exploratory data analysis
- Understand customer credit and repayment behavior
- Analyze the target class distribution
- Engineer meaningful financial and behavioral features
- Build leakage-safe preprocessing pipelines
- Compare multiple machine learning algorithms
- Handle class imbalance
- Use stratified cross-validation
- Tune the strongest-performing model
- Evaluate the final model on unseen test data
- Analyze classification thresholds
- Identify important predictive features
- Save the complete trained machine learning pipeline
- Prepare the model for web deployment using Streamlit

---

## 📊 Dataset

The dataset contains customer-level credit information, demographic attributes, historical repayment status, historical bill amounts, and previous payment amounts.

### Target Variable

`default`

| Value | Meaning |
|---|---|
| `0` | No Default |
| `1` | Default |

### Dataset Features

#### Customer Information

- `SEX`
- `EDUCATION`
- `MARRIAGE`
- `AGE`

#### Credit Information

- `LIMIT_BAL`

#### Historical Repayment Status

- `PAY_1`
- `PAY_2`
- `PAY_3`
- `PAY_4`
- `PAY_5`
- `PAY_6`

#### Historical Bill Amounts

- `BILL_AMT1`
- `BILL_AMT2`
- `BILL_AMT3`
- `BILL_AMT4`
- `BILL_AMT5`
- `BILL_AMT6`

#### Historical Payment Amounts

- `PAY_AMT1`
- `PAY_AMT2`
- `PAY_AMT3`
- `PAY_AMT4`
- `PAY_AMT5`
- `PAY_AMT6`

#### Engineered Features

- `avg_bill`
- `avg_payment`
- `credit_utilization`
- `delay_count`

The engineered features were created to capture additional customer-level financial and repayment behavior.

---

## 🔍 Exploratory Data Analysis

The dataset was explored to understand its structure, data quality, distributions, and relationships between variables.

The analysis included:

- Dataset shape and structure
- Data types
- Descriptive statistics
- Missing-value analysis
- Duplicate analysis
- Target-class distribution
- Numerical feature distributions
- Outlier analysis
- Correlation analysis
- Categorical feature analysis
- Relationship between customer behavior and default

The analysis helped identify important patterns before model development.

---

## 🛠️ Feature Engineering

Four additional features were created from the existing financial and repayment variables:

| Feature | Description |
|---|---|
| `avg_bill` | Average historical bill amount |
| `avg_payment` | Average historical payment amount |
| `credit_utilization` | Indicator of credit usage relative to available credit |
| `delay_count` | Number of historical periods with payment delays |

These features provide additional behavioral signals that may help the model distinguish between default and non-default customers.

---

## 🧹 Data Preparation

The target variable was converted into a binary classification format:

- `0` → No Default
- `1` → Default

The customer identifier column was removed because it does not provide meaningful predictive information.

The dataset was divided into training and testing sets using an 80/20 split.

Stratification was applied during the split to maintain a similar class distribution in both datasets.

---

## ⚖️ Class Imbalance

The target variable is imbalanced.

Approximate distribution:

| Class | Percentage |
|---|---:|
| No Default | 77.88% |
| Default | 22.12% |

Because default customers represent the minority class, accuracy alone is not an appropriate metric for evaluating the model.

The project therefore focuses on:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score

Class imbalance was addressed using:

- `class_weight="balanced"` for selected models
- SMOTE for selected models

SMOTE was applied inside the machine learning pipeline so that oversampling occurs only during model training and does not leak information from validation or test data.

---

## ⚙️ Data Preprocessing

### Numerical Features

Numerical features were standardized for models that are sensitive to feature scale.

`StandardScaler` was used for:

- Logistic Regression
- KNN

### Categorical Features

The following categorical features were transformed using One-Hot Encoding:

- `SEX`
- `EDUCATION`
- `MARRIAGE`

`handle_unknown="ignore"` was used to safely handle unseen categories.

### Tree-Based Models

Numerical features were passed through without scaling for:

- Decision Tree
- Random Forest
- XGBoost

Tree-based algorithms do not require numerical feature scaling.

### Pipeline Architecture

`ColumnTransformer` and `Pipeline` were used to combine preprocessing and model training into a single workflow.

This helps maintain consistency and reduce the risk of data leakage.

---

## 🔄 Machine Learning Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Train / Test Split
   ↓
Preprocessing
   ↓
Baseline Model
   ↓
Multiple Classification Models
   ↓
Stratified 5-Fold Cross-Validation
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Final XGBoost Model
   ↓
Test Set Evaluation
   ↓
Confusion Matrix
   ↓
ROC Curve
   ↓
Precision-Recall Curve
   ↓
Threshold Analysis
   ↓
Feature Importance
   ↓
Model Serialization
   ↓
Deployment

```
### Business Insights

## Repayment behavior is important
- Historical repayment and payment-delay behavior provide strong predictive signals for future default.
## False negatives are important
- The model missed 825 actual defaulters at the 0.50 threshold.
- In a real credit-risk environment, reducing these missed defaults could be financially important.
## Threshold selection matters
- Lowering the probability threshold increases recall and reduces false negatives but also increases false positives.
## Accuracy is not enough
- Because the dataset is imbalanced, a model can achieve high accuracy while still failing to identify a substantial number of default customers.
## XGBoost performed best overall
- Among the evaluated models, tuned XGBoost achieved the strongest ROC-AUC and PR-AUC performance.

  ### Technology Stack
# Programming
- Python
# Data Analysis
- Pandas
- NumPy
# Visualization
- Matplotlib
# Machine Learning
- Scikit-learn
- XGBoost
- Imbalanced-learn
# Techniques
- Feature Engineering
- One-Hot Encoding
- Standard Scaling
- ColumnTransformer
- Pipeline
- SMOTE
- Stratified K-Fold Cross-Validation
- Hyperparameter tuning
- Evaluation
- RandomizedSearchCV
- Threshold Analysis
- Feature Importance
# Deployment
- Streamlit
