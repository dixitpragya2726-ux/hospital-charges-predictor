"""
app.py
------
Interactive Streamlit demo for the Hospital Charges Predictor.

Run with:
    streamlit run src/app.py
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_data
from model import encode_categorical, predict_charge, compare_models

st.set_page_config(page_title="Hospital Charges Predictor", page_icon="🏥", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "hospital.csv")


@st.cache_resource
def load_model():
    """Loads data, trains and compares 3 models, then caches the best one."""
    df = load_data(DATA_PATH)
    df_encoded, encoders = encode_categorical(df)
    comparison = compare_models(df_encoded)
    best_name = max(comparison, key=lambda n: comparison[n]['metrics']['r2_score'])
    best_model = comparison[best_name]['model']
    best_metrics = comparison[best_name]['metrics']
    return df, best_model, encoders, best_metrics, comparison, best_name


df, model, encoders, metrics, comparison, best_model_name = load_model()

st.sidebar.header("Model Performance")
st.sidebar.caption(f"Best model: **{best_model_name}**")
st.sidebar.metric("R² Score", f"{metrics['r2_score']:.3f}")
st.sidebar.metric("Mean Absolute Error", f"${metrics['mae']:,.2f}")
st.sidebar.metric("RMSE", f"${metrics['rmse']:,.2f}")
st.sidebar.caption(
    "R² closer to 1.0 means the model explains charges more accurately."
)

with st.sidebar.expander("Compare all 3 models"):
    comparison_table = pd.DataFrame({
        name: {
            "R²": f"{res['metrics']['r2_score']:.3f}",
            "MAE": f"${res['metrics']['mae']:,.0f}",
            "RMSE": f"${res['metrics']['rmse']:,.0f}",
        }
        for name, res in comparison.items()
    }).T
    st.table(comparison_table)

st.title("🏥 Hospital Charges Predictor")
st.write(
    "Enter your details below and get an estimated annual medical charge, "
    "based on the best-performing model trained on real insurance data."
)

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

if bmi < 18.5:
    st.caption("ℹ️ This BMI falls below the typical healthy range (18.5–24.9).")
elif bmi > 30:
    st.caption("ℹ️ This BMI falls above the typical healthy range (18.5–24.9).")

if st.button("Predict My Charges", type="primary"):
    prediction = predict_charge(
        model, encoders,
        age=age, gender=gender, bmi=bmi,
        children=children, smoker=smoker, region=region
    )

    charges = df['charges']
    low_cut, high_cut = charges.quantile(0.33), charges.quantile(0.66)

    if prediction < low_cut:
        tier, tier_color = "Low estimated cost", "🟢"
    elif prediction < high_cut:
        tier, tier_color = "Medium estimated cost", "🟡"
    else:
        tier, tier_color = "High estimated cost", "🔴"

    result_col1, result_col2 = st.columns([2, 1])
    with result_col1:
        st.metric("Estimated Annual Charge", f"${prediction:,.2f}")
    with result_col2:
        st.metric("Cost Tier", f"{tier_color} {tier}")

    if smoker == 'yes':
        st.info("💡 Smoking status is the single biggest driver of cost in this model.")

    st.subheader("How this compares")

    chart_df = df.copy()
    chart_df['group'] = chart_df['smoker'].map({'yes': 'Smoker', 'no': 'Non-smoker'})

    fig = px.scatter(
        chart_df, x='age', y='charges', color='group',
        opacity=0.5,
        color_discrete_map={'Smoker': '#EF553B', 'Non-smoker': '#636EFA'},
        labels={'age': 'Age', 'charges': 'Annual Charge ($)', 'group': ''},
        title="Your estimate against the dataset (by age and smoking status)"
    )
    fig.add_scatter(
        x=[age], y=[prediction], mode='markers',
        marker=dict(size=12, color='gold', symbol='star', line=dict(width=1, color='black')),
        name='You'
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "Model: Best of Linear Regression, Random Forest, Gradient Boosting | "
    "Features: age, gender, bmi, children, smoker, region"
)
