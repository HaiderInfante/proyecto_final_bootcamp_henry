import streamlit as st
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "modelos_machine_learning/pf-henry-404414-39010891a60e.json"
# ...

# Inicializar cliente BigQuery
client = bigquery.Client()

# Cargar datos desde BigQuery
query = "SELECT * FROM `pf-henry-404414.data_machine_learning.wb_data_machine_learning_bis`"
data_original = client.query(query).to_dataframe()
data = data_original.copy()

# Preprocesamiento: One-Hot Encoding para 'Pais'
data = pd.get_dummies(data, columns=['Pais'])

# Obtener la lista de todos los países únicos
all_countries = data_original['Pais'].unique()


# Lista de columnas numéricas (sin incluir la variable objetivo)
numeric_columns = [
    'Tasa_Natalidad', 'Emisiones_CO2', 'Educacion_obligatoria_en_anios',
    'gasto_salud_per_capita_ppp', 'Tasa_Mortalidad', 'Gasto_Salud_Gobierno',
    'gasto_salud_gobierno_per_capita_ppp', 'gasto_salud_privado_pct_gasto_salud_actual',
    'logro_educativo_secundaria_inferior_pct_poblacion_25_anios_mas', 'PIB_per_capita',
    'indice_gini', 'gasto_educacion_gobierno_pct_pib', 'Esperanza_vida_femenina',
    'Tasa_alfabetizacion_adultos', 'tasa_mortalidad_lesiones_trafico',
    'mortalidad_por___enfermedades_cardiovasculares_cancer_diabetes_enf_respiratorias_pct',
    'Mortalidad_adulta_femenina', 'Mortalidad_adulta_masculina', 'Mortalidad_infantil',
    'contaminacion_pct_poblacion_excede_oms', 'personas_saneamiento_basico_pct_poblacion',
    'Acceso_agua_potable', 'estabilidad_politica', 'brecha_pobreza_2_15_dolars_a_day',
    'prevalencia_desnutricion_pct_poblacion', 'Poblacion_rural', 'desempleo_total_ilo',
    'poblacion_urbana'
]

# Define la lista de indicadores excluyendo la variable objetivo y asegurándote de que no incluya columnas no numéricas
indicators = [col for col in numeric_columns if col != 'Esperanza_vida_total']


# Separar variables predictoras y objetivo
X = data[numeric_columns]
y = data['Esperanza_vida_total']

# Normalización de variables numéricas
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Aplicación de PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Entrenar y evaluar Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Entrenar y evaluar Perceptrón Multicapa (Red Neuronal)
nn = MLPRegressor(random_state=42)
nn.fit(X_train, y_train)
nn_pred = nn.predict(X_test)

# Entrenar y evaluar Regresión Lineal
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Comparar métricas
print("Random Forest MSE:", mean_squared_error(y_test, rf_pred))
print("Random Forest R2:", r2_score(y_test, rf_pred))
print("Red Neuronal MSE:", mean_squared_error(y_test, nn_pred))
print("Red Neuronal R2:", r2_score(y_test, nn_pred))
print("Regresión Lineal MSE:", mean_squared_error(y_test, lr_pred))
print("Regresión Lineal R2:", r2_score(y_test, lr_pred))

# Validación cruzada y ajuste de hiperparámetros para Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

# Reentrenamiento con la mejor configuración
rf_optimized = RandomForestRegressor(
    n_estimators=grid_search.best_params_['n_estimators'],
    max_depth=grid_search.best_params_['max_depth'],
    min_samples_split=grid_search.best_params_['min_samples_split'],
    random_state=42
)
rf_optimized.fit(X_train, y_train)

# Función para extrapolación de indicadores
def extrapolate_indicators(data, country, indicators, target_year):
    extrapolated_values = []
    for indicator in indicators:
        country_data = data[data['Pais'] == country]
        X = country_data['Anio'].values.reshape(-1, 1)
        y = country_data[indicator].values

        model = LinearRegression()
        model.fit(X, y)

        predicted_value = model.predict([[target_year]])[0]
        extrapolated_values.append(predicted_value)

    return np.array(extrapolated_values)

# Función para predecir esperanza de vida
def predict_life_expectancy(country, year, model, data_original, scaler, pca, indicators):
    extrapolated_values = extrapolate_indicators(data_original, country, indicators, year)

    # Aplicar las transformaciones a los valores extrapolados
    scaled_values = scaler.transform([extrapolated_values])
    pca_values = pca.transform(scaled_values)

    # Realizar la predicción
    predicted_life_expectancy = model.predict(pca_values)[0]
    return predicted_life_expectancy

from difflib import get_close_matches

# Normaliza la lista de países
all_countries_normalized = [country.lower() for country in all_countries]

def find_closest_country(input_country, countries_list, cutoff=0.6):
    """
    Encuentra la coincidencia más cercana para un país dado en la lista de países.
    `cutoff` es el umbral de similitud para considerar una coincidencia.
    """
    closest_matches = get_close_matches(input_country.lower(), countries_list, n=1, cutoff=cutoff)
    return closest_matches[0] if closest_matches else None

st.title("Predicción de Esperanza de Vida")

# Input de país
country_input = st.text_input("Ingrese el país deseado:")

# Input de año
year_input = st.text_input("Ingrese el año deseado:")

# Botón de predicción
if st.button("Predecir Esperanza de Vida"):
    # Validación de entrada y predicción
    if country_input and year_input:
        country_normalized = find_closest_country(country_input, all_countries_normalized)

        if country_normalized:
            try:
                year_input = int(year_input)
            except ValueError:
                st.error("El año ingresado no es válido.")
            else:
                country_original = all_countries[all_countries_normalized.index(country_normalized)]
                predicted_life_expectancy = predict_life_expectancy(country_original, year_input, rf_optimized, data_original, scaler, pca, indicators)
                st.success(f"La esperanza de vida del país {country_original} para el año {year_input} es: {predicted_life_expectancy}")
        else:
            st.error("País no encontrado. Por favor, intente nuevamente.")
    else:
        st.warning("Por favor, complete ambos campos.")

