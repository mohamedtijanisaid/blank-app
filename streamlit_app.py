import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Fonction de chargement des données avec un chemin relatif
def load_data(file_path):
    df = pd.read_csv(file_path)
    st.write("### Aperçu des données:")
    st.dataframe(df.head())
    return df

# Fonction de prétraitement des données
def preprocess_data(df):
    st.write("### Prétraitement des données")
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df = df.drop(columns=['Date'])
    st.write("Ajout des colonnes Year et Month et suppression de Date.")
    st.dataframe(df.head())
    return df

# Fonction d'entraînement du modèle
def train_model(df):
    st.write("### Entraînement du modèle")

    X = df.drop(columns=['Weekly_Sales'])
    y = df['Weekly_Sales']

    numeric_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month']
    categorical_features = ['Holiday_Flag']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    st.write("Modèle entraîné avec succès!")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**Erreur Quadratique Moyenne (MSE):** {mse:.2f}")
    st.write(f"**Score R2:** {r2:.2f}")
    return model

# Fonction de prédiction
def predict(model):
    st.write("### Prédiction")
    st.write("Entrez les valeurs pour effectuer une prédiction :")

    input_data = {
        'Temperature': st.number_input("Temperature", value=42.0),
        'Fuel_Price': st.number_input("Fuel Price", value=2.5),
        'CPI': st.number_input("CPI", value=211.0),
        'Unemployment': st.number_input("Unemployment", value=8.0),
        'Year': st.number_input("Year", value=2010, step=1),
        'Month': st.number_input("Month", value=2, step=1),
        'Holiday_Flag': st.selectbox("Holiday Flag", options=[0, 1])
    }

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    st.write(f"**Prédiction des ventes hebdomadaires :** {prediction:.2f}")

# Fonction principale
def main():
    st.title("Analyse des ventes Walmart")
    st.sidebar.header("Navigation")

    menu = st.sidebar.radio(
        "Choisissez une étape:",
        ["Télécharger les données", "Prétraitement", "Entraînement", "Prédiction"]
    )

    # Ajout du chemin du fichier
    data_path = os.path.join("data", "Walmart_Store_sales.csv")

    if menu == "Télécharger les données":
        if os.path.exists(data_path):
            st.session_state["df"] = load_data(data_path)
        else:
            st.write("Le fichier `Walmart_Store_sales.csv` est introuvable dans le dossier `data`.")

    elif menu == "Prétraitement":
        if "df" in st.session_state:
            st.session_state["df"] = preprocess_data(st.session_state["df"])
        else:
            st.write("Veuillez télécharger les données d'abord.")

    elif menu == "Entraînement":
        if "df" in st.session_state:
            st.session_state["model"] = train_model(st.session_state["df"])
        else:
            st.write("Veuillez effectuer le prétraitement d'abord.")

    elif menu == "Prédiction":
        if "model" in st.session_state:
            predict(st.session_state["model"])
        else:
            st.write("Veuillez entraîner un modèle d'abord.")

if __name__ == "__main__":
    main()
