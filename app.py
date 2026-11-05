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
st.set_page_config(page_title="Advanced Hybrid Neuro-Genetic Loan Predictor", page_icon="🧬", layout="wide")

st.title("🧬 Advanced Neuro-Genetic Hybrid Loan Eligibility Engine")
st.markdown("**Modules Covered:** Neural Networks (Backpropagation) + Genetic Algorithms (MOOP/Optimization) + Hybrid Systems")

# -------------------------------------------------------------------
# 1. ADVANCED SYNTHETIC DATASET GENERATION
# -------------------------------------------------------------------
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

    # Complex Non-linear Approval Logic
    approved = (
        (score > 620) & 
        (dti < 0.48) & 
        (loan < (income * 4.5 + savings * 1.5)) & 
        (employment_yrs >= 1)
    ).astype(int)

    df = pd.DataFrame({
        'Income': income,
        'LoanAmount': loan,
        'CreditScore': score,
        'DTIRatio': dti,
        'EmploymentYears': employment_yrs,
        'Savings': savings,
        'Approved': approved
    })
    return df

df = generate_data()

X = df[['Income', 'LoanAmount', 'CreditScore', 'DTIRatio', 'EmploymentYears', 'Savings']]
y = df['Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------------
# 2. GENETIC ALGORITHM FOR NEURAL NETWORK HYPERPARAMETER OPTIMIZATION
# -------------------------------------------------------------------
ACTIVATIONS = ['relu', 'tanh']
LEARNING_RATES = [0.001, 0.01, 0.05]

def evaluate_chromosome(chromosome):
    l1, l2, lr_idx, act_idx = chromosome
    try:
        clf = MLPClassifier(
            hidden_layer_sizes=(l1, l2),
            activation=ACTIVATIONS[act_idx],
            learning_rate_init=LEARNING_RATES[lr_idx],
            max_iter=150,
            random_state=42
        )
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        return accuracy_score(y_test, preds)
    except:
        return 0.0

@st.cache_resource
def run_genetic_algorithm(pop_size=8, generations=4):
    # Chromosome structure: [Layer1_Neurons (8-64), Layer2_Neurons (4-32), LR_Index, Activation_Index]
    population = [
        [random.randint(8, 64), random.randint(4, 32), random.randint(0, 2), random.randint(0, 1)]
        for _ in range(pop_size)
    ]
    
    best_fitness_history = []
    best_chromosome = None
    best_accuracy = 0.0

    for gen in range(generations):
        scores = [evaluate_chromosome(chrom) for chrom in population]
        
        for idx, score in enumerate(scores):
            if score > best_accuracy:
                best_accuracy = score
                best_chromosome = population[idx]
                
        best_fitness_history.append(best_accuracy)
        
        # Selection: Top 50% Parents
        sorted_pop = [chrom for _, chrom in sorted(zip(scores, population), reverse=True)]
        parents = sorted_pop[:pop_size // 2]
        
        # Crossover & Mutation
        next_generation = parents.copy()
        while len(next_generation) < pop_size:
            p1, p2 = random.sample(parents, 2)
            child = [p1[0], p2[1], p1[2], p2[3]] # Uniform crossover
            # Mutation (10% chance)
            if random.random() < 0.1:
                child[0] = random.randint(8, 64)
            next_generation.append(child)
            
        population = next_generation

    return best_chromosome, best_accuracy, best_fitness_history

# -------------------------------------------------------------------
# 3. SIDEBAR & GA MONITORING UI
# -------------------------------------------------------------------
st.sidebar.header("🧬 Genetic Algorithm Optimization")
run_ga = st.sidebar.button("Re-run GA Evolution")

best_chrom, best_acc, ga_history = run_genetic_algorithm()

st.sidebar.markdown(f"**Best Architecture Found:**")
st.sidebar.write(f"- Layer 1 Neurons: `{best_chrom[0]}`")
st.sidebar.write(f"- Layer 2 Neurons: `{best_chrom[1]}`")
st.sidebar.write(f"- Learning Rate: `{LEARNING_RATES[best_chrom[2]]}`")
st.sidebar.write(f"- Activation: `{ACTIVATIONS[best_chrom[3]]}`")
st.sidebar.metric(label="Optimized Test Accuracy", value=f"{best_acc * 100:.2f}%")

# Plot GA Convergence Curve
fig_ga, ax_ga = plt.subplots(figsize=(4, 2.5))
ax_ga.plot(ga_history, marker='o', color='green', linewidth=2)
ax_ga.set_title("GA Fitness Convergence")
ax_ga.set_xlabel("Generation")
ax_ga.set_ylabel("Accuracy")
ax_ga.grid(True)
st.sidebar.pyplot(fig_ga)

# -------------------------------------------------------------------
# 4. FINAL HYBRID MODEL TRAINING
# -------------------------------------------------------------------
@st.cache_resource
def get_final_model(chrom):
    final_clf = MLPClassifier(
        hidden_layer_sizes=(chrom[0], chrom[1]),
        activation=ACTIVATIONS[chrom[3]],
        learning_rate_init=LEARNING_RATES[chrom[2]],
        max_iter=300,
        random_state=42
    )
    final_clf.fit(X_train_scaled, y_train)
    return final_clf

final_model = get_final_model(best_chrom)

# -------------------------------------------------------------------
# 5. USER INPUT FORM & REAL-TIME PREDICTION
# -------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 Applicant Financial Inputs")
    income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=85000, step=5000)
    loan = st.number_input("Requested Loan Amount ($)", min_value=5000, max_value=1000000, value=200000, step=5000)
    savings = st.number_input("Savings Balance ($)", min_value=0, max_value=500000, value=25000, step=2500)
    
    score = st.slider("Credit Score (FICO)", min_value=300, max_value=850, value=720)
    dti_percent = st.slider("Debt-to-Income Ratio (%)", min_value=5, max_value=70, value=25)
    employment = st.slider("Employment History (Years)", min_value=0, max_value=40, value=5)

with col_right:
    st.subheader("📊 Execution & Model Insights")
    
    if st.button("Run Neuro-Genetic Prediction", type="primary", use_container_width=True):
        dti = dti_percent / 100.0
        raw_input = np.array([[income, loan, score, dti, employment, savings]])
        scaled_input = scaler.transform(raw_input)
        
        prediction = final_model.predict(scaled_input)[0]
        prob = final_model.predict_proba(scaled_input)[0][1]

        st.divider()
        if prediction == 1:
            st.success(f"### Result: APPROVED ✅\n**Model Confidence:** {prob * 100:.1f}%")
        else:
            st.error(f"### Result: REJECTED ❌\n**Model Confidence:** {(1 - prob) * 100:.1f}%")

        # Feature Significance Plot
        st.subheader("💡 Feature Relative Importance")
        feature_names = ['Income', 'Loan Amount', 'Credit Score', 'DTI Ratio', 'Employment', 'Savings']
        
        # Calculate feature deviations from dataset mean
        mean_scaled = np.mean(X_train_scaled, axis=0)
        input_diff = np.abs(scaled_input[0] - mean_scaled)
        
        fig_feat, ax_feat = plt.subplots(figsize=(6, 3))
        ax_feat.barh(feature_names, input_diff, color='skyblue')
        ax_feat.set_xlabel("Deviation Impact Score")
        ax_feat.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig_feat)
