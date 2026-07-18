import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score
import folium
from streamlit_folium import st_folium
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Configuración de página 
st.set_page_config(
    page_title="Dashboard Híbrido: Riesgo de Incendios CV",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados 
st.markdown("""
    <style>
    /* Estilo de la fuente general */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Degradado premium en la cabecera principal */
    .title-banner {
        background: linear-gradient(135deg, #FF4B2B 0%, #FF416C 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.2);
    }
    .title-banner h1 {
        color: white !important;
        font-weight: 700;
        margin: 0;
        font-size: 2.5rem;
    }
    .title-banner p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Contenedores tipo tarjeta con efecto glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Variaciones de tarjetas de métricas según prioridad */
    .metric-positive {
        border-left: 5px solid #2ECC71;
    }
    .metric-negative {
        border-left: 5px solid #E74C3C;
    }
    .metric-warning {
        border-left: 5px solid #F39C12;
    }
    .metric-neutral {
        border-left: 5px solid #3498DB;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #7F8C8D;
        text-transform: uppercase;
        font-weight: 600;
    }
    
    /* Botones y pestañas mejoradas */
    div.stButton > button {
        background-color: #FF4B2B;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #FF416C;
        color: white;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# Banner de título superior
st.markdown("""
    <div class="title-banner">
        <h1>Sistema Inteligente de Gestión de Riesgo de Incendios Forestales</h1>
        <p>Dashboard Híbrido Multimodelo: Comparación Reactiva entre Alternativas Metodológicas</p>
    </div>
""", unsafe_allow_html=True)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datos" / "processed"

# Mapeo de nombres legibles para variables 
var_nombres = {
    'lat': 'Latitud',
    'lon': 'Longitud',
    'elevacion': 'Elevación (m)',
    'pendiente': 'Pendiente (%)',
    'orientacion': 'Orientación (grados)',
    'temp_max': 'Temp. Máxima (°C)',
    'temp_media': 'Temp. Media (°C)',
    'temp_min': 'Temp. Mínima (°C)',
    'precipitacion': 'Precipitación (mm)',
    'viento_medio': 'Viento Medio (km/h)',
    'racha_max': 'Racha Máxima (km/h)',
    'dir_viento': 'Dir. Viento (grados)',
    'humedad_media': 'Humedad Media (%)',
    'prec_acum_7d': 'Prec. Acumulada 7d (mm)',
    'tmax_max_7d': 'TMax Máxima 7d (°C)',
    'dias_sin_lluvia': 'Días sin Lluvia',
    'veg_Agricola': 'Vegetación Agrícola',
    'veg_Coniferas': 'Bosque Coníferas',
    'veg_Frondosas': 'Bosque Frondosas',
    'veg_Matorral': 'Matorral Forestal',
    'veg_Pastizal': 'Pastizal',
    'veg_Urbano_Antropizado': 'Urbano/Antropizado',
    'veg_Urbano_Otros': 'Urbano/Otros'
}

# 1. Cargar datos con caché dinámica
@st.cache_data
def cargar_datos(tipo_modelo):
    suffix = "" if tipo_modelo == "A" else "_altB"
    
    X_train_path = DATA_DIR / f"X_train{suffix}.csv"
    X_test_path = DATA_DIR / f"X_test{suffix}.csv"
    y_train_path = DATA_DIR / f"y_train{suffix}.csv"
    y_test_path = DATA_DIR / f"y_test{suffix}.csv"
    df_balanced_path = DATA_DIR / f"08_dataset_modelado_BALANCEADO{suffix}.csv"
    
    X_train = pd.read_csv(X_train_path, sep=';', decimal=',')
    X_test = pd.read_csv(X_test_path, sep=';', decimal=',')
    y_train = pd.read_csv(y_train_path, sep=';', decimal=',')['target']
    y_test = pd.read_csv(y_test_path, sep=';', decimal=',')['target']
    
   
    cols_bool = [col for col in X_train.columns if X_train[col].dtype == 'bool']
    for col in cols_bool:
        X_train[col] = X_train[col].astype(int)
        X_test[col] = X_test[col].astype(int)
        
   
    df_balanced = pd.read_csv(df_balanced_path, sep=';', encoding='utf-8')
    from sklearn.model_selection import train_test_split
    _, df_test_raw = train_test_split(df_balanced, test_size=0.20, stratify=df_balanced['target'], random_state=42)
    df_test_raw = df_test_raw.reset_index(drop=True)
    
   
    df_test_raw['fecha_dt'] = pd.to_datetime(df_test_raw['fecha_ini'])
    df_test_raw['anio'] = df_test_raw['fecha_dt'].dt.year
    df_test_raw['mes'] = df_test_raw['fecha_dt'].dt.month
    
    meses_es = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
        7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }
    df_test_raw['mes_nombre'] = df_test_raw['mes'].map(meses_es)
    df_test_raw['mes_anio'] = df_test_raw['mes_nombre'] + " " + df_test_raw['anio'].astype(str)
    df_test_raw['year_month_sort'] = df_test_raw['anio'] * 100 + df_test_raw['mes']
    
    return X_train, X_test, y_train, y_test, df_test_raw

# 2. Cargar/Entrenar modelo reactivo con caché
@st.cache_resource
def obtener_modelo(tipo_modelo, X_train, y_train):
    if tipo_modelo == "A":
        modelo_path = BASE_DIR / "lgbm_model.pkl"
        best_params = {
            'n_estimators': 134,
            'learning_rate': 0.056,
            'max_depth': 9,
            'num_leaves': 100,
            'min_child_samples': 17,
            'subsample': 0.70,
            'colsample_bytree': 0.69,
            'random_state': 42,
            'verbose': -1
        }
    else:
        modelo_path = BASE_DIR / "lgbm_model_altB.pkl"
        best_params = {
            'n_estimators': 150,
            'learning_rate': 0.06,
            'max_depth': 9,
            'num_leaves': 100,
            'min_child_samples': 15,
            'subsample': 0.70,
            'colsample_bytree': 0.69,
            'random_state': 42,
            'verbose': -1
        }
        
    try:
        model = joblib.load(modelo_path)
    except Exception:
        
        model = LGBMClassifier(**best_params)
        model.fit(X_train, y_train)
        try:
            joblib.dump(model, modelo_path)
        except Exception:
            pass
    return model

# 3. Inicializar Explicador SHAP con caché
@st.cache_resource
def obtener_explainer(_model):
    return shap.TreeExplainer(_model)

# 4. Calcular Valores SHAP con caché
@st.cache_data
def calcular_shap_values(_explainer, X_test_df):
    features = [col for col in X_test_df.columns if col != 'tipo_veg_display']
    return _explainer(X_test_df[features])


def obtener_nombre_vegetacion(row):
    veg_cols = {
        'veg_Agricola': 'Agrícola',
        'veg_Coniferas': 'Coníferas',
        'veg_Frondosas': 'Frondosas',
        'veg_Matorral': 'Matorral',
        'veg_Pastizal': 'Pastizal',
        'veg_Urbano_Antropizado': 'Urbano/Antropizado',
        'veg_Urbano_Otros': 'Urbano/Otros'
    }
    for col, display_name in veg_cols.items():
        if row[col] == 1:
            return display_name
    return 'Desconocida'

# Configuración de la barra lateral 
with st.sidebar:
    st.image("https://img.icons8.com/color/96/wildfire.png", width=90)
    
    st.markdown("### 🗺️ Enfoque Metodológico")
    enfoque = st.radio(
        "Seleccione el enfoque:",
        options=[
            "Alternativa A: Caso-Control Emparejado",
            "Alternativa B: Espacio-Temporal Puro"
        ],
        index=0,
        help="Alterna entre el modelo emparejado por fecha/municipio y el nuevo muestreo espacio-temporal puro."
    )
    tipo_modelo = "A" if "Alternativa A" in enfoque else "B"
    
    st.markdown("### ⚙️ Configuración de Alertas")
    
    
    try:
        X_train, X_test, y_train, y_test, df_test_raw = cargar_datos(tipo_modelo)
        model = obtener_modelo(tipo_modelo, X_train, y_train)
        
        
        explainer = obtener_explainer(model)
        shap_values = calcular_shap_values(explainer, X_test)
        
        
        y_proba = model.predict_proba(X_test)[:, 1]
        
        
        if 'tipo_veg_display' not in X_test.columns:
            X_test['tipo_veg_display'] = X_test.apply(obtener_nombre_vegetacion, axis=1)
            
        datos_cargados = True
    except Exception as e:
        st.error(f"Error al cargar datos o entrenar el modelo: {e}")
        datos_cargados = False

    if datos_cargados:
        umbral = st.selectbox(
            "Umbral de Decisión (Protección Civil):",
            [0.50, 0.40, 0.30, 0.20],
            index=2, # Por defecto 0.30
            help="Determina el nivel de probabilidad a partir del cual se activa la alerta de incendio."
        )
        
        
        st.markdown("---")
        st.markdown("### 📅 Escenario Temporal")
        sorted_df = df_test_raw.sort_values('year_month_sort')
        sorted_mes_anio = sorted_df['mes_anio'].unique().tolist()
        opciones_tiempo = ["Ver Todo el Histórico"] + sorted_mes_anio
        
        seleccion_tiempo = st.selectbox(
            "Ventana Temporal (Escenario Histórico):",
            opciones_tiempo,
            index=0,
            help="Filtra los datos del mapa para evaluar un escenario atmosférico homogéneo."
        )
        

        st.markdown("---")
        st.markdown("##### 💡 Nota Operativa")
        if umbral == 0.50:
            st.warning("⚠️ **Umbral Estándar (0.50)**: Equilibrio teórico óptimo, pero asume un riesgo elevado al omitir incendios reales por causas humanas imprevistas.")
        elif umbral == 0.30:
            st.success("🛡️ **Umbral Recomendado (0.30)**: Optimizado para Protección Civil. Prioriza el Recall protegiendo el territorio contra focos omitidos.")
        elif umbral == 0.20:
            st.error("🚨 **Umbral de Emergencia Extrema (0.20)**: Máxima sensibilidad, pero incrementa drásticamente las falsas alarmas, pudiendo saturar las brigadas.")
    else:
        umbral = 0.50
        seleccion_tiempo = "Ver Todo el Histórico"
        
    st.markdown("---")
    st.markdown("**TFG de Ingeniería Forestal / Ciencia de Datos**")
    st.caption("Comunidad Valenciana - Edición 2026")


if datos_cargados:
    if seleccion_tiempo != "Ver Todo el Histórico":
        filas_filtradas = df_test_raw[df_test_raw['mes_anio'] == seleccion_tiempo].index.tolist()
    else:
        filas_filtradas = list(range(len(X_test)))
        
    
    X_test_filtered = X_test.iloc[filas_filtradas]
    y_test_filtered = y_test.iloc[filas_filtradas]
    y_proba_filtered = y_proba[filas_filtradas]
    
    
    y_pred_filtered = (y_proba_filtered >= umbral).astype(int)
    
    if len(y_test_filtered) > 0:
        cm = confusion_matrix(y_test_filtered, y_pred_filtered, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f2 = fbeta_score(y_test_filtered, y_pred_filtered, beta=2)
    else:
        tn, fp, fn, tp = (0, 0, 0, 0)
        recall, precision, f2 = (0.0, 0.0, 0.0)
        
    omitidos = fn
    falsas_alarmas = fp

# Pestañas principales
tab_mapa, tab_shap, tab_metricas = st.tabs([
    "🗺️ Mapa de Riesgo y Zonas Críticas",
    "🔍 Auditoría Local SHAP 'A la Carta'",
    "📊 Rendimiento y Calibración"
])

# Pestaña 1: Mapa
with tab_mapa:
    st.subheader(f"Mapa de Riesgo en Tiempo Real ({enfoque})")
    
    if datos_cargados:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card metric-positive">
                    <div class="metric-title">Recall (Sensibilidad)</div>
                    <div class="metric-value">{recall*100:.1f}%</div>
                    <div class="metric-title">Fuegos Detectados</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
                <div class="metric-card metric-negative">
                    <div class="metric-title">Fuegos Omitidos</div>
                    <div class="metric-value">{omitidos}</div>
                    <div class="metric-title">Falsos Negativos (Peligro)</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
                <div class="metric-card metric-warning">
                    <div class="metric-title">Falsas Alarmas</div>
                    <div class="metric-value">{falsas_alarmas}</div>
                    <div class="metric-title">Falsos Positivos (Costo)</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
                <div class="metric-card metric-neutral">
                    <div class="metric-title">Eficacia F₂-Score</div>
                    <div class="metric-value">{f2:.4f}</div>
                    <div class="metric-title">Métrica Global Operativa</div>
                </div>
            """, unsafe_allow_html=True)
            
      
        lat_media = X_test_filtered['lat'].mean() if len(X_test_filtered) > 0 else X_test['lat'].mean()
        lon_media = X_test_filtered['lon'].mean() if len(X_test_filtered) > 0 else X_test['lon'].mean()
        
        m = folium.Map(
            location=[lat_media, lon_media],
            zoom_start=8,
            tiles="OpenStreetMap",
            control_scale=True
        )
        # Bucle optimizado para renderizar los registros filtrados
        for idx, row in X_test_filtered.iterrows():
            proba = y_proba[idx]
            is_conifera = row['veg_Coniferas'] == 1
            is_matorral = row['veg_Matorral'] == 1
            y_real = y_test.iloc[idx]
            
           
            if proba < umbral:
                if y_real == 1:
                    color = "#2ecc71" # Verde, pero fue fuego (Falso Negativo)
                    status_text = "Zona Segura (FALSO NEGATIVO CRÍTICO)"
                    text_color = "#E74C3C"
                    fuel_status = "Omitido por el Clima"
                else:
                    color = "#2ecc71"       # Verde (Zona Segura Real)
                    status_text = "Segura"
                    text_color = "#2ecc71"
                    fuel_status = "Bajo Peligro"
            elif proba >= umbral and (is_conifera or is_matorral):
                color = "#8b0000"       # Rojo Oscuro (Zona Crítica)
                status_text = "CRÍTICA (Desbroce Prioritario)"
                text_color = "#8b0000"
                fuel_status = "Alta Inflamabilidad (Combustible)"
            else:
                color = "#e67e22"       # Naranja (Alerta Climática/Relieve)
                status_text = "Alerta de Riesgo"
                text_color = "#e67e22"
                fuel_status = "Combustible Moderado"
                
            
            popup_html = f"""
            <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 11px; color: #2c3e50; width: 220px; line-height: 1.4;">
                <h5 style="margin: 0 0 8px 0; color: #e74c3c; font-weight: 700; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; text-transform: uppercase;">
                    Ficha de Auditoría #{idx}
                </h5>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="font-weight: 600; padding: 3px 0;">Coordenadas:</td>
                        <td style="text-align: right; padding: 3px 0; font-family: monospace;">{row['lat']:.5f}, {row['lon']:.5f}</td>
                    </tr>
                    <tr>
                        <td style="font-weight: 600; padding: 3px 0;">Realidad:</td>
                        <td style="text-align: right; padding: 3px 0; font-weight: 700; color: {'#E74C3C' if y_real==1 else '#2ECC71'};">
                            {'INCENDIO (1)' if y_real==1 else 'AUSENCIA (0)'}
                        </td>
                    </tr>
                    <tr>
                        <td style="font-weight: 600; padding: 3px 0;">Probabilidad:</td>
                        <td style="text-align: right; padding: 3px 0; font-weight: 700; color: {text_color}; font-size: 12px;">{proba*100:.2f}%</td>
                    </tr>
                    <tr>
                        <td style="font-weight: 600; padding: 3px 0;">Vegetación:</td>
                        <td style="text-align: right; padding: 3px 0;">{row['tipo_veg_display']}</td>
                    </tr>
                    <tr>
                        <td style="font-weight: 600; padding: 3px 0;">Estado Zona:</td>
                        <td style="text-align: right; padding: 3px 0; font-weight: 700; color: {text_color};">{status_text}</td>
                    </tr>
                    <tr>
                        <td style="font-weight: 600; padding: 3px 0;">Combustible:</td>
                        <td style="text-align: right; padding: 3px 0; font-style: italic;">{fuel_status}</td>
                    </tr>
                </table>
            </div>
            """
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(m)
            
        st_folium(m, use_container_width=True, height=600)
        
    else:
        st.write("*(Cargando mapa...)*")
    
    st.markdown("""
        <div class="metric-card" style="text-align: left;">
            <h4>💡 Leyenda y Regla Operativa de Zonas:</h4>
            <ul>
                <li>🟢 <b>Verde</b>: Zona Segura (probabilidad de riesgo por debajo del umbral). *Nota: Los Falsos Negativos se camuflarán en verde por no alcanzar el umbral, demostrando físicamente la necesidad de bajar el umbral a 0.30 para activarlos en alerta.*</li>
                <li>🟠 <b>Naranja</b>: Alerta de Riesgo (probabilidad por encima del umbral por calor/sequedad/viento).</li>
                <li>🔴 <b>Rojo Oscuro</b>: Zona Crítica de Actuación (Alerta activa sobre biomasa forestal de Coníferas o Matorral). <b>Objetivo prioritario para desbroce y limpieza de montes.</b></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Pestaña 2: SHAP Local
with tab_shap:
    st.subheader(f"Auditoría Local SHAP 'A la Carta' ({enfoque})")
    
    if datos_cargados:
        st.markdown("#### ⚡ Accesos Rápidos a Casos Clave del Test")
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        
        if 'selected_idx' not in st.session_state:
            st.session_state['selected_idx'] = 10 # Default
            
        with col_btn1:
            if st.button("Caso 1: Test #10 (Verdadero Positivo)"):
                st.session_state['selected_idx'] = 10
        with col_btn2:
            if st.button("Caso 2: Test #33 (Verdadero Negativo)"):
                st.session_state['selected_idx'] = 33
        with col_btn3:
            if st.button("Caso 3: Test #73 (Fronterizo)"):
                st.session_state['selected_idx'] = 73
        with col_btn4:
            if st.button("Caso 4: Test #11 (Falso Negativo Crítico)"):
                st.session_state['selected_idx'] = 11
                
        opciones_id = filas_filtradas
        if st.session_state['selected_idx'] not in opciones_id:
            opciones_id = sorted(list(set(opciones_id + [st.session_state['selected_idx']])))
            
        indice_auditar = st.selectbox(
            "Seleccione el ID del Registro de Test a auditar:",
            options=opciones_id,
            index=opciones_id.index(st.session_state['selected_idx']) if st.session_state['selected_idx'] in opciones_id else 0
        )
        
        st.session_state['selected_idx'] = indice_auditar
        
        y_real_ind = y_test.iloc[indice_auditar]
        proba_ind = y_proba[indice_auditar]
        
        st.markdown("---")
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            realidad_str = "<b style='color:#E74C3C;'>🔥 INCENDIO (1)</b>" if y_real_ind == 1 else "<b style='color:#2ECC71;'>🟢 AUSENCIA DE FUEGO (0)</b>"
            st.markdown(f"**Realidad del Suceso:** {realidad_str}", unsafe_allow_html=True)
        with col_res2:
            st.markdown(f"**Probabilidad Predictiva:** `{(proba_ind*100):.2f}%`")
        with col_res3:
            pred_color = "#E74C3C" if proba_ind >= umbral else "#2ECC71"
            pred_txt = "RIESGO ACTIVO (Alerta)" if proba_ind >= umbral else "ZONA SEGURA (Sin Alerta)"
            st.markdown(f"**Predicción Operativa:** <b style='color:{pred_color};'>{pred_txt}</b>", unsafe_allow_html=True)
            
        st.markdown("#### 📉 Justificación del Riesgo Local (Waterfall Plot)")
        shap_val_single = shap_values[indice_auditar]
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        shap.plots.waterfall(shap_val_single, max_display=10, show=False)
        plt.title(f"Justificación del Riesgo Local (Waterfall Plot - Test #{indice_auditar})", fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        shap_dict = dict(zip(X_test.columns, shap_val_single.values))
        if 'tipo_veg_display' in shap_dict:
            del shap_dict['tipo_veg_display']
            
        impactos_positivos = {k: v for k, v in shap_dict.items() if v > 0}
        impactos_negativos = {k: v for k, v in shap_dict.items() if v < 0}
        
        top_positivos = sorted(impactos_positivos.items(), key=lambda item: item[1], reverse=True)[:3]
        top_negativos = sorted(impactos_negativos.items(), key=lambda item: item[1], reverse=False)[:3]
        
        st.markdown("---")
        col_pos, col_neg = st.columns(2)
        
        with col_pos:
            st.error("🔥 AMENAZAS (Variables que SUMAN riesgo)")
            for var, val in top_positivos:
                nombre = var_nombres.get(var, var)
                val_real = X_test.iloc[indice_auditar][var]
                val_real_str = "Sí" if (var.startswith("veg_") and val_real == 1) else ("No" if (var.startswith("veg_") and val_real == 0) else f"{val_real:.2f}")
                st.markdown(f"""
                    <div style="background-color: rgba(231, 76, 60, 0.08); border-left: 4px solid #E74C3C; padding: 12px; border-radius: 6px; margin-bottom: 8px;">
                        <span style="font-weight: 700; color: #C0392B; font-size: 0.95rem;">{nombre}</span><br>
                        Valor medido: <b>{val_real_str}</b> | Impacto SHAP: <b style="color: #E74C3C;">+{val:.4f} log-odds</b>
                    </div>
                """, unsafe_allow_html=True)
                
        with col_neg:
            st.success("🟢 ESCUDOS (Variables que RESTAN riesgo)")
            for var, val in top_negativos:
                nombre = var_nombres.get(var, var)
                val_real = X_test.iloc[indice_auditar][var]
                val_real_str = "Sí" if (var.startswith("veg_") and val_real == 1) else ("No" if (var.startswith("veg_") and val_real == 0) else f"{val_real:.2f}")
                st.markdown(f"""
                    <div style="background-color: rgba(46, 204, 113, 0.08); border-left: 4px solid #2ECC71; padding: 12px; border-radius: 6px; margin-bottom: 8px;">
                        <span style="font-weight: 700; color: #27AE60; font-size: 0.95rem;">{nombre}</span><br>
                        Valor medido: <b>{val_real_str}</b> | Impacto SHAP: <b style="color: #2ECC71;">{val:.4f} log-odds</b>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.write("*(Cargando explicabilidad SHAP...)*")

# Pestaña 3: Métricas
with tab_metricas:
    st.subheader(f"Calibración y Rendimiento Comparativo ({enfoque})")
    
    if datos_cargados:
        st.markdown(f"### Matriz de Confusión para el Umbral Seleccionado ({umbral})")
        if seleccion_tiempo != "Ver Todo el Histórico":
            st.info(f"Mostrando matriz de confusión sobre el subconjunto de **{seleccion_tiempo}** ({len(filas_filtradas)} registros).")
        else:
            st.info(f"Mostrando matriz de confusión sobre el total de registros de Test ({len(X_test)} registros).")
            
        st.markdown(f"""
            <table style="width:100%; border: 1px solid rgba(255,255,255,0.1); border-collapse: collapse; text-align: center; font-size: 1.1rem;">
                <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
                    <td style="padding: 15px; border: 1px solid rgba(255,255,255,0.1);" colspan="2" rowspan="2">Matriz de Confusión</td>
                    <td style="padding: 15px; border: 1px solid rgba(255,255,255,0.1);" colspan="2">Predicción del Modelo</td>
                </tr>
                <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
                    <td style="padding: 15px; border: 1px solid rgba(255,255,255,0.1); width: 40%;">Predice AUSENCIA (0)</td>
                    <td style="padding: 15px; border: 1px solid rgba(255,255,255,0.1); width: 40%;">Predice ALERTA (1)</td>
                </tr>
                <tr>
                    <td style="padding: 15px; border: 1px solid rgba(255,255,255,0.1); font-weight: bold; background-color: rgba(255,255,255,0.02);" rowspan="2">Realidad</td>
                    <td style="padding: 15px; border: 1px solid rgba(255,255,255,0.1); font-weight: bold; background-color: rgba(255,255,255,0.02);">Sin Fuego (0)</td>
                    <td style="padding: 15px; border: 1px solid rgba(255,255,255,0.1); color: #2ECC71; font-weight: bold;">{tn} <br><span style="font-size:0.8rem; color:#7F8C8D;">Verdaderos Negativos (Zonas Seguras)</span></td>
                    <td style="padding: 15px; border: 1px solid rgba(255,255,255,0.1); color: #F39C12; font-weight: bold;">{fp} <br><span style="font-size:0.8rem; color:#7F8C8D;">Falsos Positivos (Falsas Alarmas)</span></td>
                </tr>
                <tr>
                    <td style="padding: 15px; border: 1px solid rgba(255,255,255,0.1); font-weight: bold; background-color: rgba(255,255,255,0.02);">Incendio Real (1)</td>
                    <td style="padding: 15px; border: 1px solid rgba(255,255,255,0.1); color: #E74C3C; font-weight: bold;">{fn} <br><span style="font-size:0.8rem; color:#7F8C8D;">Falsos Negativos (Fuegos Omitidos)</span></td>
                    <td style="padding: 15px; border: 1px solid rgba(255,255,255,0.1); color: #2ECC71; font-weight: bold;">{tp} <br><span style="font-size:0.8rem; color:#7F8C8D;">Verdaderos Positivos (Fuegos Detectados)</span></td>
                </tr>
            </table>
        """, unsafe_allow_html=True)
    
        y_pred_full_50 = (y_proba >= 0.50).astype(int)
        cm_full_50 = confusion_matrix(y_test, y_pred_full_50, labels=[0, 1])
        tn_50, fp_50, fn_50, tp_50 = cm_full_50.ravel()
        rec_50 = tp_50 / (tp_50 + fn_50) if (tp_50 + fn_50) > 0 else 0.0
        
        y_pred_full_30 = (y_proba >= 0.30).astype(int)
        cm_full_30 = confusion_matrix(y_test, y_pred_full_30, labels=[0, 1])
        tn_30, fp_30, fn_30, tp_30 = cm_full_30.ravel()
        rec_30 = tp_30 / (tp_30 + fn_30) if (tp_30 + fn_30) > 0 else 0.0
        
        recuperados = fn_50 - fn_30
        
        st.markdown("---")
        st.markdown(f"""
            #### 📊 Análisis Dinámico del Barrido de Umbrales (Total del Test para {enfoque}):
            *   **Umbral 0.50**: Se omiten **{fn_50}** incendios. El Recall es del **{rec_50*100:.1f}%**. Hay **{fp_50}** falsas alarmas.
            *   **Umbral 0.30**: Se omiten **{fn_30}** incendios (¡rescatamos **{recuperados}** incendios de ser ignorados!). El Recall sube al **{rec_30*100:.1f}%**. Las falsas alarmas se sitúan en **{fp_30}** (incremento asumible por seguridad civil).
        """)
