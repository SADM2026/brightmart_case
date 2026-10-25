import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the pre-trained model
model = joblib.load('model.sav')

# Set page title and layout
st.set_page_config(page_title="Sales Predictor App", layout="centered")

st.title("📈 Advertising Sales Predictor")
st.write("Enter your planned advertising budget across different channels to predict potential Sales units.")

# Input sliders for the features
st.sidebar.header("Ad Budgets ($)")
tv_budget = st.sidebar.slider("TV Budget", min_value=0.0, max_value=350.0, value=150.0, step=1.0)
radio_budget = st.sidebar.slider("Radio Budget", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
newspaper_budget = st.sidebar.slider("Newspaper Budget", min_value=0.0, max_value=150.0, value=10.0, step=1.0)

# Layout columns to display inputs and prediction
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Campaign Budgets")
    st.write(f"**TV:** ${tv_budget:,.2f}")
    st.write(f"**Radio:** ${radio_budget:,.2f}")
    st.write(f"**Newspaper:** ${newspaper_budget:,.2f}")

# Prepare input data matching feature names used during training
input_data = pd.DataFrame({
    'TV': [tv_budget],
    'Radio': [radio_budget],
    'Newspaper': [newspaper_budget]
})

# Predict using loaded model
prediction = model.predict(input_data)[0]

with col2:
    st.subheader("Predicted Output")
    st.metric(label="Estimated Sales Units", value=f"{prediction:.2f}")

st.info("This model uses Linear Regression trained on historical Advertising campaign budgets.")
