# Credit Card Default Prediction | Machine Learning

An end-to-end machine learning project that predicts credit card default risk using customer demographics, credit information, and historical repayment behavior. The project focuses on class imbalance, model comparison, hyperparameter optimization, and evaluation using business-relevant metrics.

## 🎯 Business Problem

Credit card defaults can expose financial institutions to losses. The objective is to identify customers with a higher likelihood of default using historical customer and repayment data.

The model supports credit-risk analysis by distinguishing between default and non-default cases.

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Imbalanced-learn` · `SMOTE` · `XGBoost` · `Streamlit`

## 🔍 Machine Learning Workflow

- Performed exploratory data analysis to understand repayment patterns and target-class distribution.
- Engineered features such as average bill amount, average payment amount, credit utilization, and payment-delay count.
- Split the dataset into training and testing sets.
- Evaluated multiple classification algorithms.
- Addressed class imbalance using SMOTE within the training workflow.
- Used stratified cross-validation to preserve class proportions across folds.
- Tuned XGBoost hyperparameters using `RandomizedSearchCV`.
- Evaluated the final model on held-out test data.
- Developed a Streamlit application for interactive predictions.

## 🤖 Model Comparison and Selection

The following algorithms were explored:

| Model | Role in the Experiment |
|---|---|
| Logistic Regression | Baseline model for comparison |
| K-Nearest Neighbors | Distance-based classification approach |
| Decision Tree | Nonlinear, rule-based classification |
| Random Forest | Ensemble model to capture nonlinear relationships |
| XGBoost | Gradient-boosting model selected for further tuning |

Random Forest and XGBoost were promising candidates during model comparison.

### Why was Random Forest tuning computationally expensive?

Random Forest trains multiple decision trees. Searching across many combinations of parameters—such as the number of trees, tree depth, and feature selection—can require substantial computation.

Cross-validation multiplies that cost because each candidate configuration must be evaluated across multiple folds.

Given the available computational resources, I prioritized tuning XGBoost rather than running an equally extensive Random Forest search.

### Why XGBoost?

XGBoost builds trees sequentially, with each new tree helping correct errors made by the existing ensemble. It can model nonlinear relationships and interactions between customer features.

It was selected for further hyperparameter optimization based on the earlier model-comparison results and computational practicality.

This selection does not establish that XGBoost is universally better than Random Forest; a fair comparison would require comparable tuning budgets and evaluation procedures.

## ⚖️ Handling Class Imbalance

Credit-default datasets may contain significantly more non-default cases than default cases. A model can achieve high accuracy while failing to identify a substantial proportion of actual defaulters.

To address this challenge, I used **SMOTE (Synthetic Minority Over-sampling Technique)** during model development.

SMOTE generates synthetic minority-class examples using neighboring observations in feature space.

**Implementation considerations:**
- Split the data before applying SMOTE.
- Apply SMOTE only to training data, never to the held-out test set.
- During cross-validation, perform SMOTE separately within each training fold using an imbalanced-learn pipeline.
- Evaluate performance using precision, recall, F1-score, ROC-AUC, and PR-AUC rather than relying on accuracy alone.

These precautions help prevent data leakage and produce more reliable evaluation results.

## 📊 Model Performance

**Tuned XGBoost — Held-out Test Set**

| Metric | Test Score |
|---|---:|
| ROC-AUC | 0.7772 |
| PR-AUC | 0.5583 |
| Precision | 64.71% |
| Recall | 38.28% |
| F1-score | 0.4811 |

### Key Findings

- **ROC-AUC of 0.7772:** The model demonstrates useful discrimination between default and non-default cases.
- **Precision of 64.71%:** Approximately 64.7% of customers predicted as defaulters were actual defaulters in the test set.
- **Recall of 38.28%:** The model identifies approximately 38.3% of actual defaulters at the evaluated threshold.
- **F1-score of 0.4811:** There is room to improve the balance between precision and recall.

The relatively low recall indicates that many actual defaulters remain undetected at the current threshold. This is an important limitation for a credit-risk application.

## 💡 Feature Importance

Feature-importance analysis indicated that historical payment-delay behavior was among the most influential predictive signals.

This suggests that previous repayment behavior provides useful information for estimating default risk. Feature importance identifies predictive associations; it does not establish causation.

## 🚀 Deployment

A Streamlit application allows users to enter customer information and obtain a model prediction.

The trained model is saved as `credit_default_xgboost.pkl`.

### Run Locally

```bash
git clone https://github.com/Gyaneshwar116/Credit-Card-Default-Prediction.git
cd Credit-Card-Default-Prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 🔮 End Steps

- Optimize the classification threshold to improve default detection.
- Compare tuned Random Forest and XGBoost under comparable computational budgets.
- Evaluate probability calibration.
- Use SHAP to explain individual predictions.
- Assess model fairness and performance across relevant customer groups.
- Add automated tests, reproducible training, and model monitoring.

