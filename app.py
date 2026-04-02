import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Churn Intelligence Dashboard", layout="wide")

st.title("📊 Customer Churn Intelligence Dashboard")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ---------------- PREPROCESSING ----------------
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# ---------------- FEATURES ----------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ---------------- TRAIN MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["📊 Dashboard", "🔮 Prediction"])

# ================= DASHBOARD =================
with tab1:

    st.subheader("📊 Churn Distribution")
    fig1 = px.pie(df, names="Churn")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📉 Tenure vs Churn")
    fig2 = px.box(df, x="Churn", y="tenure")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("💰 Monthly Charges vs Churn")
    fig3 = px.box(df, x="Churn", y="MonthlyCharges")
    st.plotly_chart(fig3, use_container_width=True)

# ================= PREDICTION =================
with tab2:

    st.subheader("🔮 Predict Customer Churn")

    input_data = X.iloc[0:1].copy()

    col1, col2 = st.columns(2)

    # -------- LEFT SIDE --------
    with col1:
        input_data["tenure"] = st.slider("Tenure (months)", 0, 72)
        input_data["MonthlyCharges"] = st.slider("Monthly Charges", 0, 150)
        input_data["TotalCharges"] = st.number_input("Total Charges", 0.0)

        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        contract_map = {"Month-to-month":0, "One year":1, "Two year":2}
        input_data["Contract"] = contract_map[contract]

        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        payment_map = {"Electronic check":0, "Mailed check":1, "Bank transfer":2, "Credit card":3}
        input_data["PaymentMethod"] = payment_map[payment]

    # -------- RIGHT SIDE --------
    with col2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        internet_map = {"DSL":0, "Fiber optic":1, "No":2}
        input_data["InternetService"] = internet_map[internet]

        input_data["OnlineSecurity"] = st.selectbox("Online Security", [0,1])
        input_data["TechSupport"] = st.selectbox("Tech Support", [0,1])

        input_data["SeniorCitizen"] = st.selectbox("Senior Citizen", [0,1])
        input_data["Partner"] = st.selectbox("Partner", [0,1])
        input_data["Dependents"] = st.selectbox("Dependents", [0,1])

    # ---------------- PREDICTION ----------------
    if st.button("Predict Churn"):

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High Risk of Churn ({prob*100:.1f}%)")
        else:
            st.success(f"✅ Customer is Safe ({(1-prob)*100:.1f}%)")

        # ---------------- GRAPH ----------------
        st.subheader("📊 Churn Risk Visualization")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={'text': "Churn Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"},
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)
