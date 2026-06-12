import os
import joblib
import streamlit as st
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
threshold = joblib.load(os.path.join(BASE_DIR, "models", "threshold.pkl"))

st.title("📊 Customer Churn Predictor")

st.write("Enter customer details:")

tenure = st.number_input("Tenure (months)", 0, 100)
monthly = st.number_input("Monthly Charges", 0.0, 200.0)
total = st.number_input("Total Charges", 0.0, 10000.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

if st.button("Predict Churn"):

    # ⚠️ IMPORTANT: keep ONLY numeric features for now
    input_df = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    prob = model.predict_proba(input_df)[:, 1][0]

    st.write(f"### Churn Probability: {prob:.2f}")

    if prob >= threshold:
        st.error("⚠️ High Risk of Churn")
    else:
        st.success("✅ Low Risk of Churn")