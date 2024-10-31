import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# App title and description
st.title("Customer Churn Prediction for Telecom")
st.write("Predict whether a telecom customer will churn based on their information.")

# User input fields
st.header("Enter Customer Details")
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=30.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
device_protection = st.selectbox("Device Protection", ["Yes", "No"])
online_backup = st.selectbox("Online Backup", ["Yes", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'gender': [1 if gender == 'Male' else 0],
    'SeniorCitizen': [senior_citizen],
    'Partner': [1 if partner == 'Yes' else 0],
    'Dependents': [1 if dependents == 'Yes' else 0],
    'tenure': [tenure],
    'PhoneService': [1 if phone_service == 'Yes' else 0],
    'MultipleLines': [1 if multiple_lines == 'Yes' else 0],
    'InternetService': [1 if internet_service == "Fiber optic" else 0],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract': [1 if contract == "Month-to-month" else (2 if contract == "One year" else 0)],
    'DeviceProtection': [1 if device_protection == "Yes" else 0],
    'OnlineBackup': [1 if online_backup == "Yes" else 0],
    'OnlineSecurity': [1 if online_security == "Yes" else 0],
    'PaperlessBilling': [1 if paperless_billing == "Yes" else 0]
})

# Make prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    churn_probability = model.predict_proba(input_data)[0][1]

    # Display the result
    st.write("## Prediction: ", "Churn" if prediction else "No Churn")
    st.write("## Churn Probability: {:.2f}%".format(churn_probability * 100))
