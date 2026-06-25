
from data_loader import load_data, summarize
from eda import (
    set_style, plot_age_distribution, plot_bmi_distribution,
    plot_charges_distribution, plot_charges_by_gender,
    plot_charges_by_region, plot_age_vs_charges, plot_bmi_vs_charges,
    print_correlations, plot_correlation_heatmap
)
from model import (
    try_parameters, fit_age_only_model, encode_categorical,
    fit_full_model, predict_charge
)

DATA_PATH = "../data/hospital.csv"


def main():
    # 1. Load data
    df = load_data(DATA_PATH)
    summarize(df)

    # 2. EDA
    set_style()
    plot_age_distribution(df)
    plot_bmi_distribution(df)
    plot_charges_distribution(df)
    plot_charges_by_gender(df)
    plot_charges_by_region(df)
    plot_age_vs_charges(df)
    plot_bmi_vs_charges(df)
    print_correlations(df)

    # 3. Manual linear regression (educational — shows how a model "learns")
    print("\n----- Manual Linear Regression Experiments -----")
    try_parameters(df, w=50, b=100)
    try_parameters(df, w=60, b=200, show=False)
    try_parameters(df, w=400, b=5000, show=False)

    # 4a. scikit-learn model, age only, non-smokers
    print("\n----- Age-Only Model (non-smokers) -----")
    fit_age_only_model(df)

    # 4b. Full multi-feature model (encode categoricals first)
    print("\n----- Full Multi-Feature Model -----")
    df_encoded, encoders = encode_categorical(df)
    plot_correlation_heatmap(df_encoded)
    full_model, metrics = fit_full_model(df_encoded)

    # 5. Example prediction for a new patient
    print("\n----- Example Prediction -----")
    example_charge = predict_charge(
        full_model, encoders,
        age=35, gender='male', bmi=28.0,
        children=2, smoker='yes', region='southeast'
    )
    print(f"Predicted charge for a 35-year-old male smoker, "
          f"BMI 28, 2 children, southeast region: ${example_charge:,.2f}")


if __name__ == "__main__":
    main()

