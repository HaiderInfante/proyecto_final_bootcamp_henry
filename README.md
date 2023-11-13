# PROYECTO FINAL - HENRY LABS

## Tabla de Contenido

- [Introducción](#introducción)
- [Objetivos](#objetivo-general)
- [Objetivos Específicos](#objetivos-específicos)
- [Alcance del Proyecto](#alcance-del-proyecto)
- [Equipo de Trabajo](#equipo-de-trabajo)
- [Stack Tecnológico](#stack-tecnológico)
- [Explicación y Uso del Repositorio](#explicación-del-repositorio)
- [Desarrollo General del Proyecto](#desarrollo-general-del-proyecto)
    - [EDA (Análisis Exploratorio de Datos)](#eda-análisis-exploratorio-de-datos)
    - [Planteamiento KPI'S](#planteamiento-kpis)
    - [Implementación ETL Automatizado en la Nube con GCP](#implementación-etl-automatizado-en-la-nube-con-gcp)
    - [Generación de Dashboards en Looker Studio](#generación-de-dashboards-en-looker-studio)
    - [Implementación Modelo de Machine Learning](#implementación-modelo-de-machine-learning)
- [Documentación Oficial del Proyecto](#documentación-oficial-del-proyecto)
- [Video Oficial del Proyecto](#video-oficial-del-proyecto)

## Introducción
Somos una consultora de datos llamada LatAm-Data Consulting encargada de realizar proyectos de data science sobre cualquier ámbito o sector que las empresas públicas o privadas deseen desarrollar con el fin de brindar herramientas para que tomen las mejores decisiones empresariales o corporativas basadas en datos (data-driven), estas decisiones contribuirán aumentar la eficiencia en todos los procesos con los que cuente la empresa (predicciones y pronóstico, medición de rendimientos, identificar oportunidades de negocio, análisis de competencia, reducción de riesgos, experiencia al cliente e innovación).

Para el presente proyecto el propósito es trabajar en colaboración con entidades gubernamentales para mejorar la calidad de vida de las personas, aumentar los niveles de esperanza de vida y fomentar la salud y el bienestar a nivel global. Ésto se realizará mediante un proyecto de data science completo en donde se involucren procesos de data engineering, data analytics y machine learning; basados principalmente en un dataset del Banco Mundial y otras fuentes de interés que proporcionen datos de calidad con el fin de realizar un ciclo de vida de dato completo y llegar a la resolución de los objetivos planteados.

## Objetivo General
Desarrollar un servicio de consultoría a los gobiernos de los países que conforman la OEA en donde se propongan políticas públicas con el fin de aumentar la esperanza de vida de la población analizando las variables más significativas en el sector socio-económico.

## Objetivos Específicos
* Realizar un análisis exploratorio completo de los datos del proyecto.
* Documentar claramente los pasos tomados en el análisis, incluyendo gráficos y visualizaciones relevantes.
* Desarrollar un dashboard funcional e interactivo en power bi que facilite la exploración de datos.
* Incluir filtros en power bi que permitan a los usuarios seleccionar y analizar datos específicos.
* Diseñar el dashboard el power bi de manera clara y estética para facilitar la interpretación de la información.
* Medir y graficar la reducción del 10% en la tasa de homicidios en siniestros viales de los últimos seis meses en CABA en comparación con el semestre anterior.
* Medir y graficar la reducción del 7% en la cantidad de accidentes mortales de motociclistas en el último año en CABA en comparación con el año anterior.
* Proponer, medir y graficar un tercer KPI relevante para la temática de seguridad vial.
* Garantizar que los KPIs estén representados adecuadamente en el dashboard de power bi.

## Alcance del Proyecto
Para el presente proyecto abordaremos los países que conforman la Organización de Estados Americanos (OEA), dejando para un proyecto posterior el análisis de un proyecto que englobe países de diferentes continentes. Adicional el rango de años seleccionado para el proyecto es de 32 años (1990-2022).

A través de un proceso de selección, se pre seleccionarán primero más de 50 indicadores de los sectores de las bases más confiables de los organismos internacionales especializados en el tema, que tienen que ver con el crecimiento económico, demográfico y con el desarrollo de los países, de los sectores salud, educación, economía, hábitos y bienestar, medio ambiente, nivel de vida, y condiciones financieras, entre otros.

Luego, como resultado de filtrar los datos a través de nuestro modelo específicamente diseñado, se tendrán en cuenta únicamente las variables o indicadores más significativos de cada sector que impacten de manera contundente en la esperanza de vida de la población.

Las bases de datos que se abordarán son del Banco Mundial, Naciones Unidas, Organización Mundial de la Salud, y Cepal.

## Equipo de Trabajo
Contamos con un excelente equipo de profesionales con amplios conocimientos en el campo de análisis de datos.
* Brenda Schutt, (Data Analytics, Data Scientist)
* Mara Laudonia (Data Analytics, Data Scientist)
* Haider Infante Rey, (Data Engineer, Data Scientist)

![Team](imagenes/01team.png)

## Stack Tecnológico
El stack tecnológico utilizado para este proyecto fue el siguiente:

## Explicación del Repositorio

## Desarrollo General del Proyecto

### EDA (Análisis Exploratorio de Datos)

* **Estado de la base de datos:** las bases de datos son relativamente limpias y confiables en cuanto a los valores de los indicadores. Las bases iniciales que utilizamos fueron del Banco Mundial, específicamente en los datasets World Development Indicators (WEI) y Health, Nutrition and Population Indicators (HNPI) . De acuerdo a la disponibilidad de los datos y al análisis preliminar, se decidió trabajar, extraer datos y procesar los datos como series temporales del período mencionado a (1990-2022), de cada indicador que eventualmente pueda tener incidencia en la Esperanza de Vida, clasificados por país (De la región de la OEA)

* **Datos extraídos:** en el análisis preliminar pre seleccionamos más de 50 indicadores. Para determinar el criterio de selección de los factores socioeconómicos que más podían incidir en la EV nos valimos inicialmente de las tesis de Economía de David Rodriguez, y luego fuimos profundizando a medida que íbamos analizando la data, para incorporar más indicadores y fuentes externas al Banco Mundial, como la Cepal, Naciones Unidas, y la Organización Mundial de la Salud.

* **Filtros y reorganización de los datos:** hicimos un primer filtro de nulos, con indicadores que no tenían prácticamente data para casi ningún país y, aplicamos filtro de registros duplicados. Nos quedamos con aproximadamente 48 indicadores, agrupados por país y por año, para crear las series temporales. También pasamos un filtro a la base de datos por cantidad de población por país, ya que detectamos que los países con poblaciones menores a 2 millones, que generalmente eran las islas del Caribe pequeñas y medianas, evidenciaban una mayor carencia de datos, que podían alterar la calidad de la data obtenida. Nos quedamos entonces con la lista de países de las OEA con una población mayor a 2 millones de habitantes. 

* **Detección de Nulos:** luego hicimos una nueva detección de nulos, en este caso pivotando los datos de las tablas extraídas, para analizar los nulos de las series de tiempo de los indicadores de cada país. Esta forma de presentar los datos nos permitió comprobar que la mayoría de los nulos (o ceros), tenían que ver más que nada con falta de datos en el principio de la serie, que se denotaron en indicadores de desarrollo más específicos, o bien al final de la serie, por falta de carga de datos. Pero en ningún caso el umbral de nulos superó el 25% y por tanto dejamos los nulos y los registros para luego hacer tratamiento, ya que la data disponible, aunque dispar en los registros, es valiosa para obtener información y relaciones en cada país

* **Outliers:** se buscaron outliers, por  indicador y por país. Se observa que la mayoría de los outliers no son por mala entrada de carga de datos, sino por falta de datos, en algunos años, por ejemplo, el número cero es con frecuencia un outlier, que aparece a principio de cada serie de indicador por país, evidenciando la falta de algunos indicadores sociales antes del año 2000 (en la mayoría de los casos donde se detectaron outliers). 
En otros casos, evidenciaron una eventual particularidad de un país como una crisis económica con un salto en algún indicador (por ejemplo en el 2001, el dato de desempleo para Argentina, cuando ocurrió la crisis y el default del 2001).
En otros, claramente algún error -aunque muy pocos), como ser la EV de República Dominicana, que se sitúa cercana a los 140 años.
Se decidió por tanto, dejar los outliers para no eliminar los registros y luego reemplazarlos por data más cercana a la realidad,  rellenar los nulos con el cero, para eventuales cálculos posteriores.

* **Tratamientos de nulos y outliers:** luego, en una segunda instancia, se decidió rellenar los outliers y los nulos utilizando la técnica de un modelo de regresión, por tratarse de series de tiempo. Esto lo hicimos así, porque consideramos que desde que existe la data, puede explicar fuertemente el futuro modelo de los mayores factores socioeconómicos que inciden en la esperanza de vida para ese país en particular.

* **Gráficos:** comenzamos con gráficas las tendencias de los indicadores. Luego graficamos en un boxplot los outliers, y realizamos la regresión lineal para reemplazar los outliers y luego graficamos las tendencias de los indicadores, con los datos nuevos corregidos por el modelo.
Realizamos las regresiones para los 50 indicadores por cada país analizado, en total 35.

* **Correlaciones:** hicimos una tabla de correlaciones con las series temporales corregidas por el modelo de regresión  y graficamos las correlaciones que existen entre cada indicador (50) con la esperanza de vida, por país. Luego acotamos el análisis a los indicadores que tienen las 15 correlaciones más altas con la esperanza de vida, por país, con el objetivo de determinar las primeras relaciones que nos permitan determinar los factores socioeconómicos que más inciden en la esperanza de Vida de los países analizados de la OEA.

### Planteamiento KPI'S

### Implementación ETL Automatizado en la Nube con GCP

### Generación de Dashboards en Looker Studio

### Implementación Modelo de Machine Learning

## Documentación Oficial del Proyecto
[Documentación Oficial](https://docs.google.com/document/d/1tasQgqrHd8O3r5we7FaN1J7Qnps3nfe9UmtfKFxx0hs/edit?usp=sharing)

## Video Oficial del Proyecto