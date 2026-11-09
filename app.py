import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Page Configuration
st.set_page_config(page_title="Enterprise Neuro-Fuzzy-Genetic Decision System", page_icon="🧬", layout="wide")

st.title("🧬 Multi-Objective Neuro-Fuzzy-Genetic Loan Decision Engine")
st.markdown("**Syllabus Coverage:** Module 1 (Hard vs. Soft), Module 2 (Backpropagation MLP), Module 3 (Fuzzy Logic FIS), & Module 4 (Genetic Algorithms / MOOP)")

# -------------------------------------------------------------------
# 1. SYNTHETIC DATASET GENERATION
# -------------------------------------------------------------------
@st.cache_data
def generate_complex_dataset():
    np.random.seed(42)
    n_samples = 1500
    
    income = np.random.randint(20000, 200000, n_samples)
    loan = np.random.randint(5000, 600000, n_samples)
    score = np.random.randint(300, 850, n_samples)
    dti = np.random.uniform(0.05, 0.65, n_samples)
    employment = np.random.randint(0, 30, n_samples)
    savings = np.random.randint(1000, 150000, n_samples)

    # Complex Non-Linear Decision Target
    approved = (
        (score > 620) & 
        (dti < 0.48) & 
        (loan < (income * 4.2 + savings * 1.2)) &
        (employment >= 1)
    ).astype(int)

    # Add 4% noise for soft computing imprecision test
    noise = np.random.choice(n_samples, size=int(n_samples * 0.04), replace=False)
    approved[noise] = 1 - approved[noise]

    return pd.DataFrame({
        'Income': income, 'LoanAmount': loan, 'CreditScore': score,
        'DTIRatio': dti, 'EmploymentYears': employment, 'Savings': savings,
        'Approved': approved
    })

df = generate_complex_dataset()
X = df[['Income', 'LoanAmount', 'CreditScore', 'DTIRatio', 'EmploymentYears', 'Savings']]
y = df['Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------------
# 2. MODULE 1: HARD COMPUTING ENGINE
# -------------------------------------------------------------------
def hard_computing_classifier(income, loan, score, dti, employment):
    # Strict binary thresholding with crisp rules
    if score >= 650 and dti <= 0.40 and loan <= (income * 4.0) and employment >= 2:
        return 1
    return 0

# -------------------------------------------------------------------
# 3. MODULE 3: FUZZY LOGIC ENGINE (MAMDANI FIS SIMULATION)
# -------------------------------------------------------------------
def triangular_mf(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-5), (c - x) / (c - b + 1e-5)))

def fuzzy_evaluator(score, dti):
    # Membership Functions
    poor_score = triangular_mf(score, 300, 300, 600)
    good_score = triangular_mf(score, 550, 700, 850)
    high_dti = triangular_mf(dti, 0.35, 0.65, 0.65)
    low_dti = triangular_mf(dti, 0.05, 0.05, 0.40)
    
    # Fuzzy Rules & Mamdani Inference
    rule1_weight = min(good_score, low_dti)   # High approval strength
    rule2_weight = min(poor_score, high_dti)  # High rejection strength
    
    # Defuzzification (Centroid Method Approximation)
    defuzzified_risk = (rule2_weight * 0.2 + rule1_weight * 0.8) / (rule1_weight + rule2_weight + 1e-5)
    return defuzzified_risk

# -------------------------------------------------------------------
# 4. MODULE 4: MULTI-OBJECTIVE GENETIC ALGORITHM (NSGA-II CONCEPT)
# -------------------------------------------------------------------
ACTIVATIONS = ['relu', 'tanh']
LEARNING_RATES = [0.001, 0.01, 0.05]

def evaluate_chromosome(chrom):
    l1, l2, lr_idx, act_idx = chrom
    try:
        clf = MLPClassifier(hidden_layer_sizes=(l1, l2), activation=ACTIVATIONS[act_idx],
                            learning_rate_init=LEARNING_RATES[lr_idx], max_iter=150, random_state=42)
        clf.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_scaled))
        complexity = l1 + l2  # Objective 2: Model Complexity (minimize)
        return acc, complexity
    except:
        return 0.0, 999

@st.cache_resource
def run_moea_ga():
    # Chromosome: [Layer1_Nodes, Layer2_Nodes, LR_Index, Activation_Index]
    pop = [[random.randint(8, 64), random.randint(4, 32), random.randint(0, 2), random.randint(0, 1)] for _ in range(10)]
    pareto_front = []
    
    for gen in range(4):
        evals = [evaluate_chromosome(c) for c in pop]
        # Sort by Pareto dominance: Maximize Accuracy (eval[0]), Minimize Complexity (eval[1])
        sorted_pop = sorted(zip(evals, pop), key=lambda x: (-x[0][0], x[0][1]))
        
        pareto_front = [(res[0], res[1], chrom) for res, chrom in sorted_pop[:5]]
        
        parents = [chrom for _, chrom in sorted_pop[:4]]
        # Crossover & Mutation
        pop = parents + [
            [parents[0][0], parents[1][1], parents[0][2], parents[1][3]],
            [parents[1][0], parents[0][1], parents[1][2], parents[0][3]],
            [random.randint(8, 64), parents[0][1], parents[0][2], parents[0][3]],
            [parents[1][0], random.randint(4, 32), parents[1][2], parents[1][3]],
            [random.randint(8, 64), random.randint(4, 32), random.randint(0, 2), random.randint(0, 1)],
            [parents[2][0], parents[2][1], parents[2][2], parents[2][3]]
        ]
        
    return pareto_front

pareto_front = run_moea_ga()
best_accuracy, best_complexity, best_chrom = pareto_front[0]

# Train Final MLP Model based on GA Selection
@st.cache_resource
def get_final_mlp(chrom):
    clf = MLPClassifier(hidden_layer_sizes=(chrom[0], chrom[1]), activation=ACTIVATIONS[chrom[3]],
                         learning_rate_init=LEARNING_RATES[chrom[2]], max_iter=300, random_state=42)
    clf.fit(X_train_scaled, y_train)
    return clf

mlp_model = get_final_mlp(best_chrom)

# -------------------------------------------------------------------
# 5. SIDEBAR: GA & FUZZY SYSTEM MONITORING
# -------------------------------------------------------------------
st.sidebar.title("🧬 System Architecture")
st.sidebar.markdown("### Genetic Evolution Optimal Parameters:")
st.sidebar.metric("GA-Optimized Accuracy", f"{best_accuracy * 100:.2f}%")
st.sidebar.write(f"- Network Nodes: `{best_chrom[0]} x {best_chrom[1]}`")
st.sidebar.write(f"- Learning Rate ($\eta$): `{LEARNING_RATES[best_chrom[2]]}`")
st.sidebar.write(f"- Activation: `{ACTIVATIONS[best_chrom[3]]}`")

# Plot Pareto Optimal Front
st.sidebar.subheader("Multi-Objective Pareto Front")
fig_p, ax_p = plt.subplots(figsize=(4, 2.5))
accs = [p[0] * 100 for p in pareto_front]
comps = [p[1] for p in pareto_front]
ax_p.scatter(comps, accs, color='red', zorder=3)
ax_p.plot(comps, accs, linestyle='--', color='gray')
ax_p.set_xlabel("Complexity (Neurons)")
ax_p.set_ylabel("Accuracy (%)")
ax_p.grid(True, linestyle='--', alpha=0.6)
st.sidebar.pyplot(fig_p)

# -------------------------------------------------------------------
# 6. MAIN APPLICATION FRONTEND
# -------------------------------------------------------------------
col_in, col_out = st.columns([1, 1.2])

with col_in:
    st.subheader("📋 Financial Profile Inputs")
    income = st.number_input("Annual Income ($)", 10000, 500000, 85000, 5000)
    loan = st.number_input("Requested Loan Amount ($)", 5000, 1000000, 210000, 5000)
    savings = st.number_input("Total Savings ($)", 0, 500000, 30000, 2500)
    score = st.slider("Credit Score (FICO)", 300, 850, 650)
    dti_pct = st.slider("Debt-to-Income Ratio (%)", 5, 70, 32)
    employment = st.slider("Employment History (Years)", 0, 40, 3)

with col_out:
    st.subheader("📊 Multi-Engine Inference Analysis")
    
    if st.button("Evaluate Across All Paradigms", type="primary", use_container_width=True):
        dti = dti_pct / 100.0
        
        # 1. Hard Computing Evaluation
        hard_res = hard_computing_classifier(income, loan, score, dti, employment)
        
        # 2. Fuzzy Logic Evaluation
        fuzzy_score = fuzzy_evaluator(score, dti)
        
        # 3. Neural Network Backpropagation Evaluation
        raw_in = np.array([[income, loan, score, dti, employment, savings]])
        scaled_in = scaler.transform(raw_in)
        mlp_pred = mlp_model.predict(scaled_in)[0]
        mlp_prob = mlp_model.predict_proba(scaled_in)[0][1]

        st.divider()
        
        # Output Grid
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 1. Hard Computing")
            st.info("APPROVED ✅" if hard_res == 1 else "REJECTED ❌")
            st.caption("Deterministic IF-THEN Rules")
            
        with c2:
            st.markdown("#### 2. Fuzzy Logic FIS")
            st.warning(f"Risk Score: {fuzzy_score:.2f}")
            st.caption("Centroid Defuzzification")

        with c3:
            st.markdown("#### 3. MLP Neural Net")
            if mlp_pred == 1:
                st.success(f"APPROVED ✅ ({mlp_prob*100:.1f}%)")
            else:
                st.error(f"REJECTED ❌ ({(1-mlp_prob)*100:.1f}%)")
            st.caption("GA-Backpropagation Engine")

        # Remediation Advice Engine
        if mlp_pred == 0:
            st.subheader("💡 Counterfactual Remediation Plan")
            st.warning("To transition this candidate profile to an **APPROVED** state:")
            if score < 660:
                st.write(f"- Boost Credit Score by **+{660 - score} points** (Target: 660).")
            if loan > (income * 3.8):
                st.write(f"- Reduce requested loan by **${int(loan - income * 3.8):,}**.")
            if dti_pct > 30:
                st.write(f"- Lower monthly debts to bring DTI under **30%**.")

        # Display Fuzzy Membership Curves
        st.subheader("📈 Fuzzy Logic Membership Functions")
        fig_f, ax_f = plt.subplots(figsize=(6, 2.5))
        x_score = np.linspace(300, 850, 200)
        ax_f.plot(x_score, [triangular_mf(x, 300, 300, 600) for x in x_score], label="Poor Score", color="red")
        ax_f.plot(x_score, [triangular_mf(x, 550, 700, 850) for x in x_score], label="Good Score", color="green")
        ax_f.axvline(score, color='blue', linestyle='--', label=f"Current ({score})")
        ax_f.set_title("Credit Score Fuzzy Sets")
        ax_f.legend()
        ax_f.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_f)
