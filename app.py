import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, recall_score, f1_score

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Churn Intelligence Dashboard", layout="wide")

st.title("📊 Customer Churn Intelligence Dashboard")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ---------------- PREPROCESSING ----------------
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# ---------------- FEATURES ----------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- SMOTE ----------------
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ---------------- MODEL ----------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- PREDICTION ----------------
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.4).astype(int)

# ---------------- METRICS ----------------
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Prediction", "📌 Insights"])

# ================= DASHBOARD =================
with tab1:

    st.subheader("📈 Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy*100:.2f}%")
    col2.metric("Recall", f"{recall*100:.2f}%")
    col3.metric("F1 Score", f"{f1*100:.2f}%")

    # ---------------- CHURN DISTRIBUTION ----------------
    st.subheader("📊 Churn Distribution")
    fig1 = px.pie(df, names="Churn")
    st.plotly_chart(fig1, use_container_width=True)

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("🔍 Feature Importance")

    feat_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig2 = px.bar(feat_df, x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig2, use_container_width=True)

    # ---------------- TENURE VS CHURN ----------------
    st.subheader("📉 Tenure vs Churn")

    fig3 = px.box(df, x="Churn", y="tenure")
    st.plotly_chart(fig3, use_container_width=True)

    # ---------------- MONTHLY CHARGES ----------------
    st.subheader("💰 Monthly Charges vs Churn")

    fig4 = px.box(df, x="Churn", y="MonthlyCharges")
    st.plotly_chart(fig4, use_container_width=True)

# ================= PREDICTION =================
with tab2:

    st.subheader("🔮 Predict Customer Churn")

    col1, col2 = st.columns(2)

    tenure = col1.slider("Tenure (months)", 0, 72)
    monthly = col2.slider("Monthly Charges", 0, 150)

    contract = st.selectbox("Contract Type", [0, 1, 2])
    internet = st.selectbox("Internet Service", [0, 1, 2])

    input_data = np.array([[tenure, monthly, contract, internet] + [0]*(X.shape[1]-4)])

    if st.button("Predict"):

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Churn")
        else:
            st.success("✅ Customer is Safe")

# ================= INSIGHTS =================
with tab3:

    st.subheader("📌 Business Insights")

    st.info("""
    - Customers with high monthly charges are more likely to churn
    - Low tenure customers show highest churn probability
    - Contract type plays a major role in retention
    - Electronic payment users tend to churn more
    """)

    st.subheader("🎯 Recommendations")

    st.success("""
    - Offer discounts to high-risk customers
    - Improve onboarding experience for new users
    - Promote long-term contracts
    - Target customers with personalized offers
    """)
