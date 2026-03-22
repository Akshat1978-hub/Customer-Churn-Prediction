import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title(" Customer Churn Prediction Dashboard")

# Load model
if not os.path.exists("model.pkl"):
    st.error(" Model file not found!")
    st.stop()

model = pickle.load(open("model.pkl", "rb"))

st.subheader("Enter Customer Details")

# Layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure", 0, 72)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict"):

    # Manual Encoding (FULL 19 FEATURES)
    data = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": SeniorCitizen,
        "Partner": 1 if Partner == "Yes" else 0,
        "Dependents": 1 if Dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if PhoneService == "Yes" else 0,

        "MultipleLines": 0 if MultipleLines == "No" else (1 if MultipleLines == "Yes" else 2),

        "InternetService": 0 if InternetService == "DSL" else (1 if InternetService == "Fiber optic" else 2),

        "OnlineSecurity": 0 if OnlineSecurity == "No" else (1 if OnlineSecurity == "Yes" else 2),
        "OnlineBackup": 0 if OnlineBackup == "No" else (1 if OnlineBackup == "Yes" else 2),
        "DeviceProtection": 0 if DeviceProtection == "No" else (1 if DeviceProtection == "Yes" else 2),
        "TechSupport": 0 if TechSupport == "No" else (1 if TechSupport == "Yes" else 2),

        "StreamingTV": 0 if StreamingTV == "No" else (1 if StreamingTV == "Yes" else 2),
        "StreamingMovies": 0 if StreamingMovies == "No" else (1 if StreamingMovies == "Yes" else 2),

        "Contract": 0 if Contract == "Month-to-month" else (1 if Contract == "One year" else 2),

        "PaperlessBilling": 1 if PaperlessBilling == "Yes" else 0,

        "PaymentMethod": {
            "Electronic check": 0,
            "Mailed check": 1,
            "Bank transfer (automatic)": 2,
            "Credit card (automatic)": 3
        }[PaymentMethod],

        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    input_df = pd.DataFrame([data])

    try:
        prediction = model.predict(input_df)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0]
        else:
            prob = [1 - prediction[0], prediction[0]]

        st.subheader(" Prediction Result")

        if prediction[0] == 1:
            st.error(" Customer will churn")
        else:
            st.success(" Customer will not churn")

        #  Probability Graph
        st.subheader(" Churn Probability")

        fig, ax = plt.subplots()
        labels = ["No Churn", "Churn"]
        values = prob

        ax.bar(labels, values)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")

        st.pyplot(fig)

        # Customer Profile
        st.subheader(" Customer Profile")
        st.bar_chart(input_df.T)

    except Exception as e:
        st.error(f"Error: {e}")