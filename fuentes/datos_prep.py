
# datos_prep.py
# Preparacion de datos para el proyecto Bank Marketing

import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def cargar_datos():
    """Descarga el dataset Bank Marketing desde UCI."""
    bank_marketing = fetch_ucirepo(id=222)

    X = bank_marketing.data.features.copy()
    y = bank_marketing.data.targets.copy()

    return X, y


def revisar_datos(X):
    """Muestra informacion general y valores nulos."""
    print("Dimensiones:", X.shape)
    print("\nTipos de datos:")
    print(X.dtypes)
    print("\nValores nulos por columna:")
    print(X.isnull().sum())


def preparar_datos(X, y):
    """Prepara las variables y divide los datos."""

    X_prep = X.copy()
    y_prep = y.copy()

    # Codificar la variable objetivo
    y_prep["y"] = y_prep["y"].apply(
        lambda val: 1 if str(val).strip().lower() == "yes" else 0
    )
    y_numeric = y_prep["y"].values

    # Identificar columnas numericas y categoricas
    numeric_cols = X_prep.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_cols = X_prep.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Estandarizar y codificar variables
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             categorical_cols)
        ]
    )

    # Dividir entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_prep,
        y_numeric,
        test_size=0.2,
        random_state=42,
        stratify=y_numeric
    )

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    X, y = cargar_datos()
    revisar_datos(X)

    X_train, X_test, y_train, y_test, preprocessor = preparar_datos(X, y)

    print("\nPreparacion completada.")
    print("Registros de entrenamiento:", len(X_train))
    print("Registros de prueba:", len(X_test))

