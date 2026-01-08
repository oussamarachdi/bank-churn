import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import hashlib
import os

# Page config
st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    model_path = "model/churn_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None
    return joblib.load(model_path)

model = load_model()

# Header
st.title("🏦 Bank Churn Prediction")
st.markdown("Enter customer details to predict the probability of churn.")

# Input form
with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        tenure = st.slider("Tenure (Years)", 0, 10, 5)
        balance = st.number_input("Balance", min_value=0.0, value=50000.0)
        
    with col2:
        num_products = st.slider("Number of Products", 1, 4, 1)
        has_cr_card = st.checkbox("Has Credit Card", value=True)
        is_active_member = st.checkbox("Is Active Member", value=True)
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

    submit_button = st.form_submit_button("Predict Churn Risk")

if submit_button and model:
    # Preprocess inputs
    geography_germany = 1 if geography == "Germany" else 0
    geography_spain = 1 if geography == "Spain" else 0
    
    input_data = np.array([[
        credit_score,
        age,
        tenure,
        balance,
        num_products,
        int(has_cr_card),
        int(is_active_member),
        estimated_salary,
        geography_germany,
        geography_spain
    ]])
    
    # Predict
    proba = model.predict_proba(input_data)[0, 1]
    prediction = int(proba > 0.5)
    
    # Display results
    st.markdown("---")
    st.subheader("Prediction Results")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.metric("Churn Probability", f"{proba:.2%}")
        
    with col_res2:
        if proba < 0.3:
            st.success("Risk Level: Low")
        elif proba < 0.7:
            st.warning("Risk Level: Medium")
        else:
            st.error("Risk Level: High")
            
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

    # Debug info (optional)
    with st.expander("See Input Data"):
        st.json({
            "CreditScore": credit_score,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active_member,
            "EstimatedSalary": estimated_salary,
            "Geography_Germany": geography_germany,
            "Geography_Spain": geography_spain
        })
