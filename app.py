!pip install streamlit pandas numpy scikit-learn matplotlib pyngrok



%%writefile app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Page Configuration
st.set_page_config(page_title="Loan Eligibility Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Soft Computing: Loan Eligibility Predictor")
st.markdown("**Module:** Neural Networks (Multi-Layer Perceptron trained with Backpropagation)")

# Train Model with Streamlit Caching
@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 1000

    income = np.random.randint(20000, 150000, n_samples)
    loan = np.random.randint(5000, 500000, n_samples)
    score = np.random.randint(300, 850, n_samples)
    dti = np.random.uniform(0.1, 0.6, n_samples)

    approved = ((score > 650) & (dti < 0.45) & (loan < income * 4)).astype(int)

    df = pd.DataFrame({
        'Income': income,
        'LoanAmount': loan,
        'CreditScore': score,
        'DTIRatio': dti,
        'Approved': approved
    })

    X = df[['Income', 'LoanAmount', 'CreditScore', 'DTIRatio']]
    y = df['Approved']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation='relu',
        solver='adam',
        learning_rate_init=0.01,
        max_iter=300,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    return scaler, model, acc

scaler, model, accuracy = train_model()

st.sidebar.header("Model Performance")
st.sidebar.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")

# Plot Backpropagation Loss Curve
if st.sidebar.checkbox("Show Loss Curve"):
    st.sidebar.subheader("Backpropagation Loss Curve")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(model.loss_curve_, color='blue', linewidth=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (Error)')
    ax.grid(True)
    st.sidebar.pyplot(fig)

# User Input Form
st.subheader("📋 Enter Applicant Details")
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=75000, step=5000)
    loan = st.number_input("Requested Loan Amount ($)", min_value=1000, max_value=1000000, value=150000, step=5000)

with col2:
    score = st.slider("Credit Score (FICO)", min_value=300, max_value=850, value=700)
    dti_percent = st.slider("Debt-to-Income Ratio (%)", min_value=10, max_value=70, value=30)
    dti = dti_percent / 100.0

# Live Prediction Button
if st.button("Predict Eligibility", type="primary"):
    raw_input = np.array([[income, loan, score, dti]])
    scaled_input = scaler.transform(raw_input)

    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    st.divider()
    if prediction == 1:
        st.success(f"✅ **APPROVED** — Approval Confidence: {prob * 100:.1f}%")
    else:
        st.error(f"❌ **REJECTED** — Approval Confidence: {prob * 100:.1f}%")


Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.

⠼⠴⠦Need to install the following packages:
localtunnel@2.0.2
Ok to proceed? (y) 2026-11-05 14:34:18.627 Uvicorn server started on :::8501

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://172.28.0.12:8501
  External URL: http://136.109.127.80:8501
