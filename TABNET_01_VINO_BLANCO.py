# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:54:59 2024

@author: Jaime
"""
#==============================================================================
# MODELO TABNET APICADO AL DATASET DE CALIDAD DEL VINO BLANCO
#==============================================================================
# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

# Ajustes generales de estilo para las gráficas
sns.set(style="whitegrid")

# Paso 1: Cargar el dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
df = pd.read_csv(url, sep=';')

# Asumimos que la variable 'type' está presente en el dataset. Si no, debemos agregarla de alguna manera.
# Aquí solo se trabaja con un dataset, así que se puede crear una variable 'type' para demostración
df['type'] = 'white'  # Ya que este dataset es de vino blanco

# Mostrar las primeras filas del dataset para verificar que se ha cargado correctamente
print("Primeras filas del dataset:")
print(df.head())

# Mostrar la información general del dataset para entender la estructura de los datos
print("\nInformación del dataset:")
print(df.info())

# Paso 2: Pre-procesar los Datos

# Separar las variables independientes (X) y la variable dependiente (y)
X = df.drop(columns=['quality'])
y = df['quality']

# Separar las variables numéricas y categóricas
X_numeric = X.select_dtypes(include=[np.number])
X_categorical = X.select_dtypes(exclude=[np.number])

# Estandarizar las características numéricas
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# Combinar las características numéricas estandarizadas y las categóricas
X_processed = np.concatenate([X_numeric_scaled, pd.get_dummies(X_categorical).values], axis=1)

# Dividir el dataset en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

# Paso 3: Entrenar el Modelo TabNet

# Convertir los datos a tensores
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.values.reshape(-1, 1).astype(np.float32)
y_test = y_test.values.reshape(-1, 1).astype(np.float32)

# Definir el modelo TabNet
tabnet_model = TabNetRegressor()

# Entrenar el modelo
tabnet_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric=['rmse'],
    max_epochs=100,
    patience=10,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# Evaluar el modelo en el conjunto de prueba
y_pred = tabnet_model.predict(X_test)

# Evaluar el rendimiento del modelo con métricas adicionales
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")

# Visualizar la comparación de valores reales vs predichos
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Comparación de Valores Reales y Predichos')
plt.show()

# Distribución de los residuos
plt.figure(figsize=(10, 6))
sns.histplot(y_test - y_pred, kde=True, bins=30)
plt.title('Distribución de los Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frecuencia')
plt.show()
