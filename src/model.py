"""
model.py
--------
Linear regression: from a manual, from-scratch model (to build intuition)
to a proper scikit-learn model trained on all features.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# The dataset's 'charges' values are in US Dollars (USD), since it's based
# on US healthcare cost data. This is the conversion rate used to display
# results in Indian Rupees (INR) instead. Update this value periodically
# to keep it close to the real exchange rate.
USD_TO_INR = 95.0


def usd_to_inr(usd_amount):
    """Converts a USD amount to INR for display purposes."""
    return usd_amount * USD_TO_INR


# ---------------------------------------------------------------------------
# Part 1: Manual linear regression (single feature) — for learning purposes
# ---------------------------------------------------------------------------

def estimate_charges(age, w, b):
    """y = w*x + b — the simplest possible linear model."""
    return w * age + b


def rmse(targets, predictions):
    """Root Mean Squared Error between actual and predicted values."""
    return np.sqrt(np.mean(np.square(targets - predictions)))


def try_parameters(df, w, b, show=True):
    """
    Plots the manual line y = w*age + b against actual charges,
    and prints the RMSE loss for that choice of w and b.
    """
    ages = df['age']
    targets = df['charges']
    predictions = estimate_charges(ages, w, b)

    loss = rmse(targets, predictions)
    print(f"w={w}, b={b}  ->  RMSE Loss: {loss:,.2f}")

    if show:
        plt.plot(ages, predictions, 'r', alpha=0.9)
        plt.scatter(ages, targets, s=8, alpha=0.8)
        plt.xlabel('Age')
        plt.ylabel('Charges')
        plt.legend(['Prediction', 'Actual'])
        plt.title(f'Manual Linear Fit (w={w}, b={b})')
        plt.show()

    return loss


# ---------------------------------------------------------------------------
# Part 2: scikit-learn LinearRegression — single feature (non-smokers, age)
# ---------------------------------------------------------------------------

def fit_age_only_model(df):
    """
    Fits a LinearRegression model using only 'age' as input,
    restricted to non-smokers (matches the original analysis).
    """
    non_smoker_df = df[df['smoker'] == 'no']

    inputs = non_smoker_df[['age']]
    targets = non_smoker_df['charges']

    model = LinearRegression()
    model.fit(inputs, targets)

    predictions = model.predict(inputs)
    loss = rmse(targets, predictions)

    print(f"Non-smoker, age-only model -> w={model.coef_[0]:.2f}, "
          f"b={model.intercept_:.2f}, RMSE={loss:,.2f}")

    return model


# ---------------------------------------------------------------------------
# Part 3: Full multi-feature model — the actual "production" model
# ---------------------------------------------------------------------------

def encode_categorical(df):
    """
    Label-encodes gender, smoker, and region so they can be used
    as numeric inputs to the model. Returns a NEW dataframe (does
    not mutate the original) along with the fitted encoders, so you
    can decode predictions later if needed.
    """
    df_encoded = df.copy()
    encoders = {}

    for col in ['gender', 'smoker', 'region']:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

    return df_encoded, encoders


def fit_full_model(df_encoded, test_size=0.2, random_state=42):
    """
    Trains a LinearRegression model on all available features
    (age, gender, bmi, children, smoker, region) and evaluates it
    on a held-out test set.
    """
    feature_cols = ['age', 'gender', 'bmi', 'children', 'smoker', 'region']
    X = df_encoded[feature_cols]
    y = df_encoded['charges']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        'r2_score': r2_score(y_test, predictions),
        'mae': mean_absolute_error(y_test, predictions),
        'rmse': rmse(y_test, predictions),
    }

    print("---- Full Model Performance (on test set) ----")
    print(f"R² Score: {metrics['r2_score']:.3f}")
    print(f"MAE:      ₹{usd_to_inr(metrics['mae']):,.2f}  (${metrics['mae']:,.2f})")
    print(f"RMSE:     ₹{usd_to_inr(metrics['rmse']):,.2f}  (${metrics['rmse']:,.2f})")

    return model, metrics


def compare_models(df_encoded, test_size=0.2, random_state=42):
    """
    Trains Linear Regression, Random Forest, and Gradient Boosting on the
    same train/test split, and returns a comparison of their performance.
    """
    feature_cols = ['age', 'gender', 'bmi', 'children', 'smoker', 'region']
    X = df_encoded[feature_cols]
    y = df_encoded['charges']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    candidates = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(random_state=random_state),
    }

    results = {}
    print("---- Model Comparison (on test set) ----")
    for name, candidate_model in candidates.items():
        candidate_model.fit(X_train, y_train)
        predictions = candidate_model.predict(X_test)

        model_metrics = {
            'r2_score': r2_score(y_test, predictions),
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': rmse(y_test, predictions),
        }
        results[name] = {'model': candidate_model, 'metrics': model_metrics}

        print(f"{name:20s} R²={model_metrics['r2_score']:.3f}  "
              f"MAE=${model_metrics['mae']:,.2f}  "
              f"RMSE=${model_metrics['rmse']:,.2f}")

    best_name = max(results, key=lambda n: results[n]['metrics']['r2_score'])
    print(f"\nBest model by R²: {best_name}")

    return results


def predict_charge(model, encoders, age, gender, bmi, children, smoker, region):
    """
    Predicts the medical charge for a single new person.
    """
    import pandas as pd

    row = pd.DataFrame([{
        'age': age,
        'gender': encoders['gender'].transform([gender])[0],
        'bmi': bmi,
        'children': children,
        'smoker': encoders['smoker'].transform([smoker])[0],
        'region': encoders['region'].transform([region])[0],
    }])

    prediction = model.predict(row)[0]
    return max(prediction, 0)
