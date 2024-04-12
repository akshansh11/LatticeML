import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xg
import scipy.stats as stats

# Streamlit app
def main():
    # Set page background color to black using custom CSS
    st.markdown("""
        <style>
            body {
                background-color: black;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title and subtitle of the app
    st.title("LatticeML")
    st.subheader("Data-Driven web application for Graph-based Architected Materials")
    
    # Display lattice.jpg image just below the title
    st.image("lattice.jpg", use_column_width=True)

    # Allow the user to upload a CSV file
    file = st.file_uploader("Upload your CSV file:", type=['csv'])

    if file:
        # Load data and preprocess it
        df = load_data(file)
        
        # Prepare data for training
        X = df.drop('Young Modulus of Architected Material (Gpa)', axis=1)
        y = df['Young Modulus of Architected Material (Gpa)']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        # Standardize the features
        sc_X = StandardScaler()
        X_train = pd.DataFrame(sc_X.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(sc_X.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        # Create and fit the model
        classifier = create_model(X_train, y_train)

        # User inputs for prediction
        st.subheader("Enter your own inputs to get a prediction:")
        input_features = {}

        for column in X.columns:
            # Use appropriate input widget based on the column type
            if df[column].dtype == 'int64' or df[column].dtype == 'float64':
                input_features[column] = st.number_input(f"Enter {column}", min_value=df[column].min(), max_value=df[column].max(), step=0.01)
            else:
                # Convert categorical features using a dropdown
                input_features[column] = st.selectbox(f"Select {column}", df[column].unique())
        
        # Convert user inputs to a DataFrame for prediction
        input_df = pd.DataFrame([input_features], columns=X.columns)
        
        # Standardize user inputs
        input_df = pd.DataFrame(sc_X.transform(input_df), columns=X.columns)
        
        # Make prediction based on user inputs
        prediction = classifier.predict(input_df)
        
        # Display the prediction
        st.subheader("Prediction")
        st.write(f"The predicted value for your inputs is: {prediction[0]:.4f}")
        
        # Calculate metrics for training
        y_pred = classifier.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display model performance metrics
        st.subheader("Model Performance")
        st.write(f"Mean Squared Error: {mse:.4f}")
        st.write(f"Mean Absolute Error: {mae:.4f}")
        st.write(f"R Squared Error: {r2:.4f}")

        # Plot feature importances
        st.subheader("Feature Importances")
        plot_feature_importances(classifier, X_train)

        # Plot actual vs. predicted values
        st.subheader("Actual vs. Predicted Plot")
        plot_actual_vs_predicted(y_test, y_pred)

        # Plot residuals
        st.subheader("Residual Plot")
        plot_residuals(y_test, y_pred)

        # Plot Q-Q plot for residuals
        st.subheader("Q-Q Plot for Residuals")
        residuals = y_test - y_pred
        plot_qq_plot(residuals)

# Function to load data and preprocess it
def load_data(file):
    df = pd.read_csv(file)
    label_encoder = LabelEncoder()
    df['Lattice Type'] = label_encoder.fit_transform(df['Lattice Type'])
    df = df.drop(columns=['Young Modulus of Alloy (Gpa)', 'Conductivity of Alloy (W/m.K)'])
    return df

# Function to create the model and fit the data
def create_model(X_train, y_train):
    classifier = xg.XGBRegressor()
    classifier.fit(X_train, y_train)
    return classifier

# Function to plot the correlation heatmap
def plot_correlation_heatmap(df):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, mask=mask, cmap='twilight', linewidths=1, annot=True, fmt=".2f", cbar=True, ax=ax)
    ax.set_title("Correlation Heatmap", fontsize=16)
    st.pyplot(fig)

# Function to plot feature importances
def plot_feature_importances(model, X_train):
    importances = model.feature_importances_
    feature_importances = pd.DataFrame(importances, index=X_train.columns, columns=['Importance'])
    top_features = feature_importances.sort_values(by='Importance', ascending=False).head(8)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y=top_features.index, data=top_features, palette='mako', ax=ax)
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.4f}", (p.get_width() + 0.01, p.get_y() + p.get_height() / 2), ha='left', va='center')
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title('Top Feature Importance', fontsize=16)
    st.pyplot(fig)

# Function to plot actual vs. predicted values
def plot_actual_vs_predicted(actual, predicted):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(actual, predicted, c=predicted, cmap='jet', marker='o', edgecolors='k', alpha=0.8)
    fig.colorbar(sc, label='Predicted', orientation='vertical')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs. Predicted Plot (XG Boost Regression)')
    ax.grid(True)
    ax.plot([min(actual), max(actual)], [min(actual), max(actual)], linestyle='--', color='gray', lw=2)
    st.pyplot(fig)

# Function to plot residuals
def plot_residuals(actual, predicted):
    residuals = actual - predicted
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(predicted, residuals, c=residuals, cmap='jet', marker='o', edgecolors='k', alpha=0.8)
    fig.colorbar(sc, label='Residuals', orientation='vertical')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot (XG Boost Regression)')
    ax.axhline(y=0, color='gray', linestyle='--', lw=2)
    st.pyplot(fig)

# Function to create a Q-Q plot for residuals
def plot_qq_plot(residuals):
    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(residuals, plot=ax, fit=True, dist='norm')
    ax.set_title('Q-Q Plot for Residuals (XG Boost Regression)')
    st.pyplot(fig)

if __name__ == '__main__':
    main()




