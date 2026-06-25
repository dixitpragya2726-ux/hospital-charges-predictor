import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def set_style():
    """Applies consistent, readable styling across all matplotlib charts."""
    sns.set_style('darkgrid')
    matplotlib.rcParams['font.size'] = 14
    matplotlib.rcParams['figure.figsize'] = (10, 6)


def plot_age_distribution(df):
    fig = px.histogram(
        df, x='age', marginal='box', nbins=47,
        title='Distribution of Age'
    )
    fig.update_layout(bargap=0.1)
    fig.show()


def plot_bmi_distribution(df):
    fig = px.histogram(
        df, x='bmi', marginal='box',
        color_discrete_sequence=['red'],
        title='Distribution of BMI (Body Mass Index)'
    )
    fig.update_layout(bargap=0.1)
    fig.show()


def plot_charges_distribution(df):
    fig = px.histogram(
        df, x='charges', marginal='box', color='smoker',
        color_discrete_sequence=['green', 'grey'],
        title='Annual Medical Charges'
    )
    fig.update_layout(bargap=0.1)
    fig.show()


def plot_charges_by_gender(df):
    fig = px.box(df, x='gender', y='charges', color='gender',
                 title='Medical Charges by Gender')
    fig.show()


def plot_charges_by_region(df):
    fig = px.box(df, x='region', y='charges', color='region',
                 title='Medical Charges by Region')
    fig.show()


def plot_age_vs_charges(df):
    fig = px.scatter(
        df, x='age', y='charges', color='smoker', opacity=0.8,
        hover_data=['gender'], title='Age vs. Charges'
    )
    fig.update_traces(marker_size=5)
    fig.show()


def plot_bmi_vs_charges(df):
    fig = px.scatter(
        df, x='bmi', y='charges', color='smoker', opacity=0.8,
        hover_data=['gender'], title='BMI vs. Charges'
    )
    fig.update_traces(marker_size=5)
    fig.show()


def print_correlations(df, smoker_numeric_col='smoker_numeric'):
    """Prints correlation of age, bmi, and smoking status with charges."""
    print("----------------------- Correlation -----------------------")
    print("Age vs Charges:    ", df['charges'].corr(df['age']))
    print("BMI vs Charges:    ", df['charges'].corr(df['bmi']))

    smoker_map = {'no': 0, 'yes': 1}
    smoker_numeric = df['smoker'].map(smoker_map)
    print("Smoker vs Charges: ", df['charges'].corr(smoker_numeric))


def plot_correlation_heatmap(df_encoded):
    """Expects a fully-numeric dataframe (after label encoding)."""
    sns.heatmap(df_encoded.corr(), cmap='Reds', annot=True)
    plt.title('Correlation Matrix')
    plt.show()