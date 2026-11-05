import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Page Configuration
st.set_page_config(page_title="Soft vs Hard Computing - Loan Predictor", page_icon="🧠", layout="wide")

st.title("🧠 Soft vs. Hard Computing: Loan Eligibility Evaluator")
st.markdown("**Syllabus Coverage:** Module 1 (Hard vs. Soft Computing) & Module 2 (Backpropagation Neural Networks)")

# -------------------------------------------------------------------
# 1. DATASET GENERATION
# -------------------------------------------------------------------
@st.cache_data
def generate_data():
    np.random.seed(42)
    n_samples = 1200
    
    income = np.random.randint(20000, 150000, n_samples)
    loan = np.random.randint(5000, 500000, n_samples)
    score = np.random.randint(300, 850, n_samples)
    dti = np.random.uniform(0.1, 0.6, n_samples)

    # Complex non-linear target variable with noise (Soft Computing Domain)
    approved = (
        (score > 630) & 
        (dti < 0.45) & 
        (loan < income * 4.2)
    ).astype(int)

    # Add 5% noise to simulate real-world imprecise data
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    approved[noise_idx] = 1 - approved[noise_idx]

    return pd.DataFrame({
        'Income': income,
        'LoanAmount': loan,
        'CreditScore': score,
        'DTIRatio': dti,
        'Approved': approved
    })

df = generate_data()
X = df[['Income', 'LoanAmount', 'CreditScore', 'DTIRatio']]
y = df['Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------------
# 2. MODULE 1: HARD COMPUTING MODEL (CRISP / RULE-BASED)
# -------------------------------------------------------------------
def hard_computing_predict(income, loan, score, dti):
    # Strict binary thresholding (Hard Computing: Crisp logic, zero tolerance for noise)
    if score >= 650 and dti <= 0.40 and loan <= (income * 4.0):
        return 1
    else:
        return 0

hard_preds = [hard_computing_predict(row['Income'], row['LoanAmount'], row['CreditScore'], row['DTIRatio']) for _, row in X_test.iterrows()]
hard_accuracy = accuracy_score(y_test, hard_preds)

# -------------------------------------------------------------------
# 3. MODULE 2: SIDEBAR NEURAL NETWORK HYPERPARAMETER TUNING
# -------------------------------------------------------------------
st.sidebar.title("⚙️ Module 2: Backpropagation Parameters")
st.sidebar.markdown("Explore factors affecting Backpropagation training:")

learning_rate = st.sidebar.select_slider(
    "Learning Rate Co-efficient (η):",
    options=[0.0001, 0.001, 0.01, 0.1, 0.5],
    value=0.01
)

activation_function = st.sidebar.selectbox(
    "Activation Function:",
    options=['relu', 'logistic', 'tanh'],
    index=0
)

hidden_layers = st.sidebar.radio(
    "Network Architecture:",
    options=["Single Layer (Perceptron: 8 nodes)", "Multilayer Perceptron (16, 8 nodes)"],
    index=1
)

hidden_structure = (8,) if "Single" in hidden_layers else (16, 8)

max_epochs = st.sidebar.slider("Max Training Epochs:", 10, 500, 200, step=10)

# Train MLP Model
@st.cache_resource
def train_soft_model(structure, act, lr, epochs):
    clf = MLPClassifier(
        hidden_layer_sizes=structure,
        activation=act,
        solver='adam',
        learning_rate_init=lr,
        max_iter=epochs,
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)
    return clf

soft_model = train_soft_model(hidden_structure, activation_function, learning_rate, max_epochs)
soft_preds = soft_model.predict(X_test_scaled)
soft_accuracy = accuracy_score(y_test, soft_preds)

# -------------------------------------------------------------------
# 4. PARADIGM COMPARISON DASHBOARD (MODULE 1)
# -------------------------------------------------------------------
st.subheader("📊 Paradigm Performance Comparison")
col_acc1, col_acc2 = st.columns(2)

with col_acc1:
    st.metric(
        label="Hard Computing (Rigid Rules)", 
        value=f"{hard_accuracy * 100:.2f}%", 
        delta="Crisp Logic"
    )

with col_acc2:
    st.metric(
        label="Soft Computing (MLP Backpropagation)", 
        value=f"{soft_accuracy * 100:.2f}%", 
        delta=f"{(soft_accuracy - hard_accuracy) * 100:+.2f}% vs Hard"
    )

st.divider()

# -------------------------------------------------------------------
# 5. INPUT FORM & DUAL PREDICTION UI
# -------------------------------------------------------------------
col_input, col_results = st.columns([1, 1.2])

with col_input:
    st.subheader("📋 Applicant Input Data")
    income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=75000, step=5000)
    loan = st.number_input("Requested Loan Amount ($)", min_value=1000, max_value=1000000, value=160000, step=5000)
    score = st.slider("Credit Score (FICO)", min_value=300, max_value=850, value=640)
    dti_percent = st.slider("Debt-to-Income Ratio (%)", min_value=10, max_value=70, value=42)
    dti = dti_percent / 100.0

with col_results:
    st.subheader("🔍 Prediction Output")
    
    if st.button("Compare Systems", type="primary", use_container_width=True):
        # Hard Computing Prediction
        hard_result = hard_computing_predict(income, loan, score, dti)
        hard_text = "APPROVED ✅" if hard_result == 1 else "REJECTED ❌"
        
        # Soft Computing Prediction
        raw_input = np.array([[income, loan, score, dti]])
        scaled_input = scaler.transform(raw_input)
        soft_result = soft_model.predict(scaled_input)[0]
        soft_prob = soft_model.predict_proba(scaled_input)[0][1]
        soft_text = "APPROVED ✅" if soft_result == 1 else "REJECTED ❌"

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown("### Hard Computing")
            st.info(f"**Status:** {hard_text}\n\n*Deterministic IF-THEN logic*")
        
        with res_col2:
            st.markdown("### Soft Computing")
            if soft_result == 1:
                st.success(f"**Status:** {soft_text}\n\n**Confidence:** {soft_prob * 100:.1f}%")
            else:
                st.error(f"**Status:** {soft_text}\n\n**Confidence:** {(1 - soft_prob) * 100:.1f}%")

        # -------------------------------------------------------------------
        # 6. MODULE 2: BACKPROPAGATION LOSS CURVE
        # -------------------------------------------------------------------
        st.subheader("📈 Backpropagation Loss Curve")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(soft_model.loss_curve_, color='#1f77b4', linewidth=2)
        ax.set_title(f"Gradient Descent Convergence (η={learning_rate}, Act={activation_function})")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Training Error (Loss)")
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
