# CHANGELOG — Actividad 5

## Versión 1.0.0 — 19 de octubre de 2026

### Descripción general

Desarrollo de un proyecto de clasificación binaria utilizando el dataset Bank Marketing del repositorio UCI Machine Learning.

Se implementaron dos algoritmos de aprendizaje automático: Regresión Logística y Random Forest.

### Incorporaciones

- Creación del repositorio Actividad-5 en GitHub.
- Organización de las carpetas `datos`, `fuentes` y `evidencias`.
- Incorporación del notebook `entrena.ipynb`.
- Creación del script `datos_prep.py` para la preparación de datos.
- Creación del script `train.py` para entrenamiento y evaluación.
- Implementación de GridSearchCV con validación cruzada estratificada de cinco particiones.
- Cálculo de Accuracy, Precision, Recall, F1-score y ROC-AUC.
- Integración de MLflow para el seguimiento de experimentos.
- Exportación de resultados a `resultados_mlflow.csv`.
- Generación de la gráfica `comparacion_modelos.png`.
- Documentación del proyecto en `README.md`.

### Correcciones

- Corrección de la serialización de modelos en MLflow mediante `cloudpickle`.
- Corrección de errores de indentación en `train.py`.
- Verificación de sintaxis mediante `py_compile`.
- Comprobación del entrenamiento de ambos modelos desde Google Colab.

### Resultados principales

| Métrica | Regresión Logística | Random Forest |
|---|---:|---:|
| Accuracy | 0.9005 | 0.9062 |
| Precision | 0.6411 | 0.6762 |
| Recall | 0.3393 | 0.3809 |
| F1-score | 0.4438 | 0.4873 |
| ROC-AUC | 0.9061 | 0.9281 |

Random Forest presentó un desempeño superior en las métricas evaluadas.

### Limitaciones identificadas

- Los experimentos de MLflow se ejecutaron inicialmente en un entorno temporal de Google Colab.
- La interfaz web de MLflow presentó restricciones de seguridad y conectividad.
- Ambos modelos muestran oportunidades de mejora en la recuperación de casos positivos.
- Se requiere fortalecer el versionamiento de dependencias y la persistencia de artefactos.

### Próximas mejoras

- Incorporar pruebas automatizadas.
- Evaluar técnicas para tratar el desequilibrio entre clases.
- Automatizar la ejecución de experimentos.
- Establecer almacenamiento persistente para MLflow.
- Monitorear posibles cambios en la distribución de los datos.

## Conclusión

El proyecto permitió implementar un flujo de entrenamiento y evaluación reproducible a partir de código versionado en GitHub.

Se comprobó el registro de métricas mediante MLflow y se generaron evidencias comparativas de los algoritmos.

La integración de herramientas de versionamiento y seguimiento experimental constituye una base para mejorar las prácticas de MLOps.
