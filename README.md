\# Modelos de Machine Learning para la Predicción de Susceptibilidad de Incendios Forestales en la Comunitat Valenciana







Este repositorio contiene el ecosistema técnico, los cuadernos de experimentación y los módulos de código fuente desarrollados para el Trabajo Fin de Grado (TFG) en Ciencia de Datos.







\## 🚀 Estructura del Proyecto







\*   `src/`: Módulos de Python para la descarga automática de datos (AEMET, Open-Meteo), procesamiento geoespacial (COSCV 2024, SRTM 30m) y generación de pseudo-ausencias.



\*   `src/alternativa\_b/`: Scripts específicos para el enfoque de validación espacio-temporal.



\*   `notebooks/`: Cuadernos Jupyter ordenados cronológicamente que cubren desde la limpieza inicial hasta la interpretación de modelos con TreeSHAP.



\*   `datos/`: Directorios destinados al almacenamiento de datos crudos (EGIF/MITECO) y matrices procesadas (excluidos del control de versiones por volumen).







## Ejecución del Proyecto

El sistema está diseñado en **tres fases desacopladas**: un pipeline de

adquisición de datos (`src/`), un ciclo de experimentación y entrenamiento

(`notebooks/`) y un dashboard de despliegue (`src/dashboard.py`).

Los modelos preentrenados (`.pkl`) están incluidos directamente en el

repositorio, lo que permite evaluar la aplicación al instante sin necesidad

de procesar datos ni reentrenar ningún modelo.

---

### Camino A — Despliegue Inmediato 

Este es el camino rápido. Los modelos LightGBM optimizados y las matrices

de Test ya están disponibles en el repositorio. Siga estos **3 pasos** para

ver el dashboard en funcionamiento en menos de 2 minutos:

**Paso 1 — Clonar el repositorio**

git clone https://github.com/marcoavgustoo/tfg-incendios-valencia.git

cd tfg-incendios-valencia



Paso 2 — Instalar dependencias

Se recomienda hacerlo dentro de un entorno virtual de Conda o venv:

# Opción A: entorno Conda (recomendado)

conda env create -f environment.yml

conda activate ads

# Opción B: pip estándar

pip install -r requirements.txt



Paso 3 — Lanzar el dashboard

streamlit run src/dashboard.py



La aplicación se abrirá automáticamente en http://localhost:8501. No se requiere ningún paso adicional: el dashboard leerá los modelos lgbm_model.pkl y lgbm_model_altB.pkl de la raíz del repositorio y las matrices de Test de datos/processed/.



Camino B — Reconstrucción Completa

Este camino permite reproducir desde cero la totalidad del experimento: desde la descarga de datos meteorológicos hasta el reentrenamiento de los modelos. Requiere una API key de AEMET OpenData (gratuita) guardada en datos/APIs/api_AEMET.txt.



Fase 1 · Pipeline de Adquisición y Enriquecimiento de Datos

Ejecute los siguientes scripts en orden estricto. Cada uno lee la salida del anterior desde datos/processed/:

# 1.1 Orquestación inicial: limpieza del registro EGIF (MITECO) y

#     cruce con la red de estaciones AEMET (genera 03_dataset_final_CV.csv)

python src/run_pipeline.py

# 1.2 Imputación climática por reanálisis ERA5 vía Open-Meteo

#     (resuelve el 40% de nulos de AEMET, genera 04_dataset_completo_clima_elevacion.csv)

python src/completar_clima_era5.py

# 1.3 Extracción de humedad relativa horaria desde ERA5

#     (genera 05_dataset_final_para_EDA.csv)

python src/extraer_humedad.py

# 1.4 Integración de cobertura vegetal COSCV 2024 mediante cruce espacial

#     (genera 06_dataset_final_enriquecido.csv)

python src/extraer_vegetacion.py

# 1.5 Extracción topográfica: pendiente y orientación de ladera (SRTM 30m)

#     (genera 07_dataset_final_TOPOGRAFICO.csv)

python src/extraer_topografia_final_api.py

# 1.6 Generación de pseudo-ausencias — Alternativa A (Caso-Control Emparejado)

#     (genera 08_dataset_modelado_BALANCEADO.csv)

python src/generar_ausencias.py

# 1.7 Pipeline completo para la Alternativa B (Muestreo Espacio-Temporal Puro)

python src/alternativa_b/generar_ausencias_altB.py

python src/alternativa_b/completar_clima_era5_altB.py

python src/alternativa_b/extraer_topografia_final_api_altB.py

Tiempo estimado: La fase completa puede tardar entre 2 y 4 horas en función de la latencia de las APIs externas (Open-Meteo, Open Topo Data). Los scripts incluyen reintentos automáticos y cachés en disco en datos/external/cache_aemet/ para poder reanudar si se interrumpe la ejecución.



Fase 2 · Preprocesamiento, Entrenamiento y Generación de Modelos

El entrenamiento de los modelos se realiza de forma interactiva desde los Jupyter Notebooks de la carpeta notebooks/. Ábralos y ejecútelos en orden:

notebooks/

├── 08_Preprocesamiento_Modelado.ipynb        # OHE + partición 80/20 Alt A

├── 08b_Preprocesamiento_Modelado_altB.ipynb  # OHE + partición 80/20 Alt B

├── 09_Entrenamiento_Modelos_Base.ipynb       # Competición baseline XGBoost vs LightGBM

├── 10_Optimizacion_Hiperparametros.ipynb     # RandomizedSearchCV Alt A (50 iter × 5-Fold)

├── 10b_Optimizacion_Hiperparametros_altB.ipynb

├── 11_Calibracion_Explicabilidad_SHAP.ipynb  # Umbral F₂, SHAP global y local Alt A

└── 11b_Calibracion_Explicabilidad_SHAP_altB.ipynb

Al ejecutar los notebooks 10 y 10b, los modelos optimizados se serializan automáticamente como lgbm_model.pkl y lgbm_model_altB.pkl en la raíz del repositorio, listos para ser consumidos por el dashboard.



Fase 3 · Despliegue

Una vez disponibles los .pkl y las matrices de Test, lance el dashboard siguiendo el Paso 3 del Camino A.

Que te parece? Es profesional, es coherente? Muestra la realidad de lo que debe hacer el usuario para reproducir el trabajo? Según esto, para que sea totalmente reproducible también debemos añadir los siguientes archivos en github: las matrices de Test de datos/processed/ y  Requiere una API key de AEMET OpenData (gratuita) guardada en datos/APIs/api_AEMET.txt.
