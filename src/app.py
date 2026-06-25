import streamlit as st
import pandas as pd
from data_loader import load_data
from model import encode_categorical, fit_full_model, predict_charge

st.set_page_config(page_title="Hospital Charges Predictor", page_icon="🏥")

st.title("🏥 Hospital Charges Predictor")
st.write(
    "Enter your details below and get an estimated annual medical charge, "
    "based on a linear regression model trained on real insurance data."
)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "hospital.csv")


@st.cache_resource
def load_model():
    """Loads data and trains the model once, then caches it across reruns."""
    df = load_data(DATA_PATH)
    df_encoded, encoders = encode_categorical(df)
    model, metrics = fit_full_model(df_encoded)
    return model, encoders, metrics


model, encoders, metrics = load_model()

# Show model performance in the sidebar
st.sidebar.header("Model Performance")
st.sidebar.metric("R² Score", f"{metrics['r2_score']:.3f}")
st.sidebar.metric("Mean Absolute Error", f"${metrics['mae']:,.2f}")

st.header("Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=100, value=30)
    bmi = st.slider("BMI", min_value=10.0, max_value=55.0, value=25.0, step=0.1)
    children = st.slider("Number of Children", min_value=0, max_value=10, value=0)

with col2:
    gender = st.selectbox("Gender", encoders['gender'].classes_)
    smoker = st.selectbox("Smoker", encoders['smoker'].classes_)
    region = st.selectbox("Region", encoders['region'].classes_)

if st.button("Predict My Charges", type="primary"):
    prediction = predict_charge(
        model, encoders,
        age=age, gender=gender, bmi=bmi,
        children=children, smoker=smoker, region=region
    )
    st.success(f"### Estimated Annual Charge: ${prediction:,.2f}")

    if smoker == 'yes':
        st.info("💡 Smoking status is the single biggest driver of cost in this model.")

st.divider()
st.caption(
    "Model: Linear Regression | Features: age, gender, bmi, children, smoker, region"
)
