# 🏥 Hospital Charges Predictor

A machine learning project that predicts annual medical insurance charges based on
patient attributes such as age, gender, BMI, number of children, smoking status,
and region. Built during my ML internship to practice the full pipeline: data
cleaning, exploratory data analysis, and linear regression modeling.

🔗 **Live App:** https://hospital-charges-predictor-g92aynwnbcszvf9kjgmxwz.streamlit.app

---

## 📊 Project Overview

Healthcare costs vary widely between individuals. This project explores **what
drives that variation** and builds a model to predict charges for a new patient.

**Key question:** Can we predict a person's medical charges using simple,
known attributes — and which attribute matters most?

**Spoiler:** Smoking status turns out to be the single biggest driver of cost.

---

## 🗂️ Dataset

| Column     | Description                              |
|------------|-------------------------------------------|
| `age`      | Age of the primary beneficiary            |
| `gender`   | Male / Female                             |
| `bmi`      | Body Mass Index                           |
| `children` | Number of dependents covered by insurance |
| `smoker`   | Smoking status (yes/no)                   |
| `region`   | Residential region in the US              |
| `charges`  | Annual medical insurance cost (target)    |

---

## 🔍 Exploratory Data Analysis

Explored:
- Distribution of age, BMI, and charges
- Charges broken down by gender and region
- Relationship between age/BMI and charges, separated by smoking status
- Correlation matrix across all features

**Key findings:**
- Smokers pay dramatically higher charges than non-smokers, regardless of age or BMI.
- Charges increase with age, but the slope is much steeper for smokers.
- BMI alone is a weak predictor — its effect is amplified by smoking status.

---

## 🤖 Modeling Approach

1. **Manual linear regression** — hand-picked slope/intercept values to build
   intuition for how a model "learns," measured with RMSE.
2. **scikit-learn LinearRegression** — trained on all 6 features, evaluated with
   R², MAE, and RMSE on a held-out test set.

### Results

| Metric    | Value      |
|-----------|------------|
| R² Score  | 0.783      |
| MAE       | $4,186.51  |
| RMSE      | $5,799.59  |

---

## 🚀 Getting Started

```bash
git clone https://github.com/dixitpragya2726-ux/hospital-charges-predictor.git
cd hospital-charges-predictor
pip install -r requirements.txt
cd src
streamlit run app.py
