"""
data_loader.py
----------------
Loads and cleans the hospital charges dataset.
"""

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the raw CSV and splits/cleans it into a proper dataframe.

    Parameters
    ----------
    filepath : str
        Relative or absolute path to the hospital.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with correct column names and numeric types.
    """
    df = pd.read_csv(filepath)

    # If the file was loaded as a single column, split it
    if df.shape[1] == 1:
        df = df.iloc[:, 0].str.split(',', expand=True)
        df.columns = ['age', 'gender', 'bmi', 'children', 'smoker', 'region', 'charges']

    # Convert numeric columns
    numeric_cols = ['age', 'bmi', 'children', 'charges']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    return df


def summarize(df: pd.DataFrame) -> None:
    """Prints a quick summary of the dataframe (shape, dtypes, stats)."""
    print(df.head())
    print(f"\nRows, Columns: {df.shape}")
    print("\nColumn dtypes:")
    print(df.info())
    print("\nNumeric summary:")
    print(df.describe())