import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="centered"
)

# Load Model

model = joblib.load("credit_default_xgboost.pkl")

# Title
st.title("Credit Card Default Prediction")
st.write(
    "Predict whether a customer is likely to default on "
    "their credit card payment using Machine Learning."
)

st.divider()

# Customer Information
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    limit_bal = st.number_input("Credit Limit", min_value=0, value=50000)
    sex = st.selectbox("Sex", ['Male', 'Female'])
    education = st.selectbox("Education", ['University','Graduate school', 'High School', 'Unknown','Others',0])

with col2:
    marriage = st.selectbox("Marital Status", ['Single', 'Married', 'Other',0])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

# Repayment History
st.subheader("Repayment History")

pay_values = {}

cols = st.columns(3)

for i in range(1, 7):
    with cols[(i - 1) % 3]:
        pay_values[f"PAY_{i}"] = st.selectbox(
            f"PAY_{i}",
            [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            index=1
        )

# Bill Amounts
st.subheader("Monthly Bill Amounts")

bill_values = {}

cols = st.columns(3)

for i in range(1, 7):
    with cols[(i - 1) % 3]:
        bill_values[f"BILL_AMT{i}"] = st.number_input(
            f"BILL_AMT{i}",
            value=10000,
            step=1000
        )

# Payment Amounts
st.subheader("Monthly Payment Amounts")

payment_values = {}

cols = st.columns(3)

for i in range(1, 7):
    with cols[(i - 1) % 3]:
        payment_values[f"PAY_AMT{i}"] = st.number_input(
            f"PAY_AMT{i}",
            min_value=0,
            value=2000,
            step=500
        )

# Prediction
st.divider()

if st.button("Predict Default"):

    customer_data = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        **pay_values,
        **bill_values,
        **payment_values
    }

    input_df = pd.DataFrame([customer_data])

    # Engineered Features
    bill_columns = [f"BILL_AMT{i}" for i in range(1, 7)]
    payment_columns = [f"PAY_AMT{i}" for i in range(1, 7)]
    repayment_columns = [f"PAY_{i}" for i in range(1, 7)]

    input_df["avg_bill"] = input_df[bill_columns].mean(axis=1)
    input_df["avg_payment"] = input_df[payment_columns].mean(axis=1)

    input_df["credit_utilization"] = (
        input_df["avg_bill"]
        / input_df["LIMIT_BAL"].replace(0, float("nan"))
    ).fillna(0)

    input_df["delay_count"] = (
        input_df[repayment_columns] > 0
    ).sum(axis=1)

    feature_order = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
        "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
        "avg_bill", "avg_payment",
        "credit_utilization", "delay_count"
    ]

    input_df = input_df[feature_order]

    try:
        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            default_probability = (
                probabilities[classes.index(1)] * 100)

            no_default_probability = (
                probabilities[classes.index(0)] * 100)

            if prediction == 1:
                st.error("Default Predicted")
            else:
                st.success("No Default Predicted")

            col1, col2 = st.columns(2)

            col1.metric(
                "Default Probability",
                f"{default_probability:.2f}%")

            col2.metric(
                "No Default Probability",
                f"{no_default_probability:.2f}%")

        else:
            st.write(f"Prediction: {prediction}")

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
