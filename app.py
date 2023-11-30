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
import matplotlib.pyplot as plt
import seaborn as sns

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

st.set_option('deprecation.showfileUploaderEncoding', False)

# Lista de temas
temas = ["Dashboards KPI'S-Looker Studio", "Modelo de Machine Learning", "Preguntas Machine Learning", "Acerca de Nosotros"]

# Barra lateral con botones para cada tema
tema_seleccionado = st.sidebar.radio("Selecciona un tema", temas)

# Página principal
st.title("Presentación Proyecto Final Henry")

# Mostrar contenido según el tema seleccionado
if tema_seleccionado == "Dashboards KPI'S-Looker Studio":
    # Función para mostrar gráfica con título en negrita y centrado
    def mostrar_grafica(titulo, url):
        st.write(f"**{titulo}**")
        st.markdown(f'<div style="text-align: center;"><iframe width="900" height="750" src="https://lookerstudio.google.com/embed/reporting/2c58e29c-4ccc-43aa-a8ed-d55c4a12a9a1/page/p_k6vdcpc1bd" frameborder="0" style="border:0" allowfullscreen></iframe></div>', unsafe_allow_html=True)

    # Página principal
    st.markdown("<h2>Dashboards KPI'S-Looker Studio</h2>", unsafe_allow_html=True)  # Título más pequeño
    # Mostrar las gráficas con títulos en negrita y centrados
    mostrar_grafica("", "https://lookerstudio.google.com/embed/reporting/2c58e29c-4ccc-43aa-a8ed-d55c4a12a9a1/page/p_zalhmmsqbd")

elif tema_seleccionado == "Modelo de Machine Learning":
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

elif tema_seleccionado == "Preguntas Machine Learning":
    # Diccionario para almacenar las predicciones de esperanza de vida para cada país en 2040
    predictions_2040 = {}

    # Iterar sobre todos los países y obtener las predicciones
    for country in all_countries:
        predicted_life_expectancy = predict_life_expectancy(country, 2040, rf_optimized, data_original, scaler, pca, indicators)
        predictions_2040[country] = predicted_life_expectancy

    # Ordenar los países por su esperanza de vida predicha en 2040 de mayor a menor
    top_countries_2040 = sorted(predictions_2040.items(), key=lambda x: x[1], reverse=True)

    # Crear la aplicación Streamlit
    st.title("¿Cuales son los cinco países con la mayor esperanza de vida en 2040?")

    # Agregar un botón para mostrar la lista de los cinco países
    if st.button("Consulta #1"):
        # Imprimir los cinco países con la mayor esperanza de vida en 2040
        for i in range(5):
            st.write(f"{i+1}. {top_countries_2040[i][0]}: {top_countries_2040[i][1]} años")



    # Ordenar las predicciones en orden ascendente
    sorted_predictions = sorted(predictions_2040.items(), key=lambda x: x[1])

    # Crear la aplicación Streamlit
    st.title("¿Cuales son los cinco países con la menor esperanza de vida en 2040?")

    # Agregar un botón para mostrar la lista de los cinco países con la esperanza de vida más baja
    if st.button("Consulta #2"):    
        # Imprimir los cinco países con la esperanza de vida más baja en 2040
        for country, life_expectancy in sorted_predictions[:5]:
            st.write(f"{country}: {life_expectancy} años")


    columns_of_interest = ['Anio', 'Esperanza_vida_total']
    data_subset_temporal = data[columns_of_interest]

    # Agrupar por año y calcular la esperanza de vida promedio
    mean_life_expectancy_by_year = data_subset_temporal.groupby('Anio')['Esperanza_vida_total'].mean().reset_index()

    # Crear la aplicación Streamlit
    st.title('Cambio en la Esperanza de Vida Global (Promedio) en las Últimas Dos Décadas')

    # Configurar el tamaño del gráfico si es necesario
    plt.figure(figsize=(10, 6))

    # Graficar el cambio en la esperanza de vida a lo largo del tiempo usando Matplotlib
    sns.lineplot(x='Anio', y='Esperanza_vida_total', data=mean_life_expectancy_by_year, marker='o')

    # Ajustar el rango del eje y para mejorar la visualización
    y_min = mean_life_expectancy_by_year['Esperanza_vida_total'].min()
    y_max = mean_life_expectancy_by_year['Esperanza_vida_total'].max()
    plt.ylim(y_min, y_max)

    # Agregar título y etiquetas al gráfico
    plt.title('Cambio en la Esperanza de Vida Global (Promedio) en las Últimas Dos Décadas')
    plt.xlabel('Año')
    plt.ylabel('Esperanza de Vida Promedio')
    plt.grid(True)

    # Mostrar el gráfico en Streamlit
    st.pyplot(plt)

elif tema_seleccionado == "Acerca de Nosotros":

    st.write("## Acerca de Nosotros")
    
    # Agrega el contenido para el Tema 3
    st.write("Somos una consultora de datos llamada LatAm-Data Consulting encargada de realizar proyectos de data science sobre cualquier ámbito o sector que las empresas públicas o privadas deseen desarrollar con el fin de brindar herramientas para que tomen las mejores decisiones empresariales o corporativas basadas en datos (data-driven), estas decisiones contribuirán aumentar la eficiencia en todos los procesos con los que cuente la empresa (predicciones y pronóstico, medición de rendimientos, identificar oportunidades de negocio, análisis de competencia, reducción de riesgos, experiencia al cliente e innovación).")
    
    st.write("Para el presente proyecto, el propósito es trabajar en colaboración con entidades gubernamentales para mejorar la calidad de vida de las personas, aumentar los niveles de esperanza de vida y fomentar la salud y el bienestar a nivel global. Esto se realizará mediante un proyecto de data science completo en donde se involucren procesos de data engineering, data analytics y machine learning; basados principalmente en un dataset del Banco Mundial y otras fuentes de interés que proporcionen datos de calidad con el fin de realizar un ciclo de vida de dato completo y llegar a la resolución de los objetivos planteados.")
    
    st.write("## Equipo de Trabajo")
    st.write("Contamos con un excelente equipo de profesionales con amplios conocimientos en el campo de análisis de datos.")
    
    # Agrega la información del equipo
    st.write("* Brenda Schutt, (Data Analytics, Data Scientist)")
    st.write("* Mara Laudonia (Data Analytics, Data Scientist)")
    st.write("* Haider Infante Rey, (Data Engineer, Data Scientist)")
    
    # Agrega la imagen del equipo
    st.image("imagenes/01team.png", caption="Equipo de Trabajo")

else:
    st.write("Selecciona un tema para ver contenido específico.")

# Fin del código
