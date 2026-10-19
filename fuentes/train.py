
# train.py
# Entrenamiento, validacion y registro en MLflow

import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from datos_prep import cargar_datos, preparar_datos


def entrenar_modelo(nombre, clasificador, parametros,
                    X_train, X_test, y_train, y_test,
                    preprocessor):

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clasificador)
    ])

    grid_params = {
        f"classifier__{clave}": valores
        for clave, valores in parametros.items()
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    with mlflow.start_run(run_name=nombre):

        busqueda = GridSearchCV(
            estimator=pipeline,
            param_grid=grid_params,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True
        )

        busqueda.fit(X_train, y_train)

        mejor_modelo = busqueda.best_estimator_

        y_pred = mejor_modelo.predict(X_test)
        y_prob = mejor_modelo.predict_proba(X_test)[:, 1]

        metricas = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, y_pred, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "cv_best_roc_auc": busqueda.best_score_
        }

        mlflow.log_params(busqueda.best_params_)
        mlflow.log_metrics(metricas)

        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        mlflow.sklearn.log_model(
            mejor_modelo,
            name="modelo"
        )

        # Guardar los resultados de validacion cruzada
        resultados_cv = pd.DataFrame(busqueda.cv_results_)
        resultados_cv.to_csv(
            "resultados_cv.csv", index=False
        )
        mlflow.log_artifact("resultados_cv.csv")
        os.remove("resultados_cv.csv")

        print(f"\nModelo: {nombre}")
        print("Mejores parametros:", busqueda.best_params_)

        for metrica, valor in metricas.items():
            print(f"{metrica}: {valor:.4f}")

        return metricas


def main():

    mlflow.set_experiment("Bank_Marketing_Classification")

    X, y = cargar_datos()

    X_train, X_test, y_train, y_test, preprocessor = (
        preparar_datos(X, y)
    )

    modelos = {
        "Logistic_Regression": (
            LogisticRegression(random_state=42),
            {
                "C": [0.1, 1.0, 10.0],
                "max_iter": [1000]
            }
        ),
        "Random_Forest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [50, 100],
                "max_depth": [10, 20]
            }
        )
    }

    for nombre, (modelo, parametros) in modelos.items():
        entrenar_modelo(
            nombre,
            modelo,
            parametros,
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor
        )


if __name__ == "__main__":
    main()
