import streamlit as st

st.set_option('deprecation.showfileUploaderEncoding', False)

# Lista de temas
temas = ["Dashboards KPI'S-Looker Studio", "Modelo de Machine Learning", "Acerca de Nosotros"]

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
    st.write("Modelo de Machine Learning")
    # Agrega aquí el contenido para el Tema 2
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
