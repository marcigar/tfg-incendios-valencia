\# Modelos de Machine Learning para la Predicción de Susceptibilidad de Incendios Forestales en la Comunitat Valenciana



Este repositorio contiene el ecosistema técnico, los cuadernos de experimentación y los módulos de código fuente desarrollados para el Trabajo Fin de Grado (TFG) en Ciencia de Datos.



\## 🚀 Estructura del Proyecto



\*   `src/`: Módulos de Python para la descarga automática de datos (AEMET, Open-Meteo), procesamiento geoespacial (COSCV 2024, SRTM 30m) y generación de pseudo-ausencias.

\*   `src/alternativa\_b/`: Scripts específicos para el enfoque de validación espacio-temporal.

\*   `notebooks/`: Cuadernos Jupyter ordenados cronológicamente que cubren desde la limpieza inicial hasta la interpretación de modelos con TreeSHAP.

\*   `datos/`: Directorios destinados al almacenamiento de datos crudos (EGIF/MITECO) y matrices procesadas (excluidos del control de versiones por volumen).



\## 🛠️ Requisitos e Instalación



El proyecto utiliza un entorno virtual de Miniconda (`ads`) para garantizar la reproducibilidad de los resultados. Para reconstruir el entorno, ejecute en su terminal:



```bash

conda env create -f environment.yml

conda activate ads 



Para replicar el procesamiento completo de los datos y desplegar la interfaz interactiva, asegúrese de tener el entorno activado y ejecute los bloques de código en el siguiente orden:

### 1. Pipeline de Procesamiento y Enriquecimiento
Debido a la migración de infraestructura meteorológica analizada en la memoria, el procesamiento se realiza ejecutando de forma secuencial la extracción inicial y los módulos de enriquecimiento geoespacial y climático:

```bash
# Paso 1: Orquestación del flujo inicial y limpieza MITECO
python src/run_pipeline.py

# Paso 2: Ingesta de reanálisis climático ERA5 e interpolación de humedad
python src/completar_clima_era5.py
python src/extraer_humedad.py

# Paso 3: Extracción de variables topográficas y botánicas (COSCV 2024)
python src/extraer_topografia_final_api.py
python src/extraer_vegetacion.py

# Paso 4: Generación de pseudo-ausencias (Ecosistema base)
python src/generar_ausencias.py 

Una vez generadas las matrices finales en la carpeta datos/processed/ (o utilizando los modelos .pkl ya preentrenados e incluidos en la raíz), puede lanzar la interfaz interactiva de Streamlit ejecutando: 

streamlit run src/dashboard.py
