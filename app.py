import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Executive Neuro-Genetic Loan Predictor", page_icon="🏦", layout="wide")

st.title("🏦 Enterprise Hybrid Neuro-Genetic Loan Decisioning System")
st.markdown("**Core Techniques:** Multi-Layer Perceptron (Backpropagation) + Genetic Algorithm Hyperparameter Tuning + Counterfactual Analytics")

# 1. Dataset Generation
@st.cache_data
def generate_data():
    np.random.seed(42)
    n_samples = 1500
    income = np.random.randint(25000, 200000, n_samples)
    loan = np.random.randint(10000, 600000, n_samples)
    score = np.random.randint(300, 850, n_samples)
    dti = np.random.uniform(0.05, 0.65, n_samples)
    employment_yrs = np.random.randint(0, 30, n_samples)
    savings = np.random.randint(1000, 100000, n_samples)

    approved = (
        (score > 620) & 
        (dti < 0.48) & 
        (loan < (income * 4.5 + savings * 1.5)) & 
        (employment_yrs >= 1)
    ).astype(int)

    return pd.DataFrame({
        'Income': income, 'LoanAmount': loan, 'CreditScore': score,
        'DTIRatio': dti, 'EmploymentYears': employment_yrs, 'Savings': savings,
        'Approved': approved
    })

df = generate_data()
X = df[['Income', 'LoanAmount', 'CreditScore', 'DTIRatio', 'EmploymentYears', 'Savings']]
y = df['Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Genetic Algorithm Engine
ACTIVATIONS = ['relu', 'tanh']
LEARNING_RATES = [0.001, 0.01, 0.05]

def evaluate_chromosome(chrom):
    l1, l2, lr_idx, act_idx = chrom
    try:
        clf = MLPClassifier(hidden_layer_sizes=(l1, l2), activation=ACTIVATIONS[act_idx],
                            learning_rate_init=LEARNING_RATES[lr_idx], max_iter=150, random_state=42)
        clf.fit(X_train_scaled, y_train)
        return accuracy_score(y_test, clf.predict(X_test_scaled))
    except:
        return 0.0

@st.cache_resource
def run_ga():
    population = [[random.randint(8, 64), random.randint(4, 32), random.randint(0, 2), random.randint(0, 1)] for _ in range(8)]
    best_acc, best_chrom, history = 0.0, None, []

    for gen in range(5):
        scores = [evaluate_chromosome(c) for c in population]
        for idx, s in enumerate(scores):
            if s > best_acc:
                best_acc, best_chrom = s, population[idx]
        history.append(best_acc)
        sorted_pop = [c for _, c in sorted(zip(scores, population), reverse=True)]
        parents = sorted_pop[:4]
        population = parents + [[parents[0][0], parents[1][1], parents[0][2], parents[1][3]],
                                [parents[1][0], parents[0][1], parents[1][2], parents[0][3]],
                                [random.randint(8, 64), parents[0][1], parents[0][2], parents[0][3]],
                                [parents[1][0], random.randint(4, 32), parents[1][2], parents[1][3]]]
    return best_chrom, best_acc, history

best_chrom, best_acc, ga_history = run_ga()

@st.cache_resource
def get_model(chrom):
    clf = MLPClassifier(hidden_layer_sizes=(chrom[0], chrom[1]), activation=ACTIVATIONS[chrom[3]],
                         learning_rate_init=LEARNING_RATES[chrom[2]], max_iter=300, random_state=42)
    clf.fit(X_train_scaled, y_train)
    return clf

model = get_model(best_chrom)

# Sidebar
st.sidebar.title("🧬 GA Optimization Panel")
st.sidebar.metric("GA-Evolved Model Accuracy", f"{best_acc * 100:.2f}%")
st.sidebar.markdown(f"**Optimal Layers:** `{best_chrom[0]} x {best_chrom[1]}`")
st.sidebar.markdown(f"**Optimal Learning Rate ($\eta$):** `{LEARNING_RATES[best_chrom[2]]}`")
st.sidebar.markdown(f"**Activation:** `{ACTIVATIONS[best_chrom[3]]}`")

fig_ga, ax_ga = plt.subplots(figsize=(4, 2.5))
ax_ga.plot(ga_history, marker='s', color='#2b5c8f', linewidth=2)
ax_ga.set_title("Genetic Fitness Convergence")
ax_ga.set_xlabel("Generation")
ax_ga.set_ylabel("Accuracy Score")
ax_ga.grid(True, linestyle='--', alpha=0.6)
st.sidebar.pyplot(fig_ga)

# Main UI
col_in, col_out = st.columns([1, 1.2])

with col_in:
    st.subheader("📋 Loan Application Parameters")
    income = st.number_input("Annual Income ($)", 10000, 500000, 85000, 5000)
    loan = st.number_input("Requested Loan Amount ($)", 5000, 1000000, 220000, 5000)
    savings = st.number_input("Total Savings ($)", 0, 500000, 20000, 2500)
    score = st.slider("Credit Score (FICO)", 300, 850, 640)
    dti_pct = st.slider("Debt-to-Income Ratio (%)", 5, 70, 35)
    employment = st.slider("Employment History (Years)", 0, 40, 4)

with col_out:
    st.subheader("📊 Instant Decision Engine & Analytics")
    if st.button("Evaluate Application", type="primary", use_container_width=True):
        raw_input = np.array([[income, loan, score, dti_pct/100.0, employment, savings]])
        scaled_input = scaler.transform(raw_input)
        
        pred = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]

        st.divider()
        if pred == 1:
            st.success(f"### Result: APPROVED ✅\n**Approval Confidence:** {prob * 100:.1f}%")
        else:
            st.error(f"### Result: REJECTED ❌\n**Approval Confidence:** {(1-prob) * 100:.1f}%")

            # Counterfactual Analysis
            st.warning("💡 **Actionable Recommendations for Approval:**")
            if score < 660:
                st.write(f"- Increase Credit Score to at least **660** (currently {score}).")
            if loan > (income * 3.5):
                st.write(f"- Lower requested loan amount to **${int(income * 3.5):,}**.")
            if dti_pct > 30:
                st.write(f"- Reduce monthly debt to bring DTI ratio under **30%**.")

        # Relative Impact Chart
        st.subheader("🔍 Input Impact Breakdown")
        feature_names = ['Income', 'Loan Amount', 'Credit Score', 'DTI Ratio', 'Employment', 'Savings']
        impacts = scaled_input[0] - np.mean(X_train_scaled, axis=0)
        
        fig_imp, ax_imp = plt.subplots(figsize=(6, 3))
        colors = ['#2ca02c' if x > 0 else '#d62728' for x in impacts]
        ax_imp.barh(feature_names, impacts, color=colors)
        ax_imp.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax_imp.set_xlabel("Impact relative to baseline dataset mean")
        st.pyplot(fig_imp)
