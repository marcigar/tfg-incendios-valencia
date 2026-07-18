"""
Script definitivo para la generación de muestras negativas.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import requests
import time
import os
from pathlib import Path
from shapely.geometry import Point
from tqdm import tqdm



BASE_DIR = Path(__file__).resolve().parent.parent
DATOS_PROCESSED = BASE_DIR / "datos" / "processed"
DATOS_EXTERNAL = BASE_DIR / "datos" / "external" / "vegetacion_cv"


CSV_INCENDIOS = DATOS_PROCESSED / "07_dataset_final_TOPOGRAFICO.csv"
GPKG_VEGETACION = DATOS_EXTERNAL / "04_2024COSCV_25830_GPKG" / "COScv2024.gpkg"
CSV_SALIDA = DATOS_PROCESSED / "08_dataset_modelado_BALANCEADO.csv"


PAUSE_OPENMETEO = 0.1  
PAUSE_OPENTOPODATA = 1.0  

# ============================================================
# 1. CARGA Y LIMPIEZA DE INCENDIOS POSITIVOS
# ============================================================

def cargar_incendios_limpios():
    """Carga los incendios reales aplicando el filtro de unicidad."""
    print(f"Leyendo dataset maestro: {CSV_INCENDIOS.name}...")
    df = pd.read_csv(CSV_INCENDIOS, sep=';', decimal=',')
    
    
    MAPEO_SUPERFICIES_CORRUPTAS = {
        "1.501.393": 150.1393,
        "10.699.999.999.999.900": 1.07,
        "11.904.680.000.000.000": 11904.68,
        "116.044.237": 11604.4237,
        "12.000.000.000.000.000": 1.2,
        "12.999.999.999.999.900": 1.3,
        "14.300.000.000.000.000": 1.43,
        "14.388.999.999.999.900": 14.389,
        "14.626.675": 1462.6675,
        "15.000.000.000.000.000": 1.5,
        "17.000.000.000.000.000": 1.7,
        "17.939.999.999.999.900": 17.94,
        "185.358.806": 18535.8806,
        "21.500.000.000.000.000": 2.15,
        "26.649.999.999.999.900": 266.5,
        "30.501.737.000.000.000": 3050.1737,
        "4.732.155": 473.2155,
        "4.752.199.999.999.990": 475.22,
        "6.255.800.000.000.000": 625.58,
        "60.600.000.000.000.000": 6.06,
        "65.440.000.000.000.000": 6.544,
        "7.100.699.999.999.990": 710.07,
        "7.333.999.999.999.990": 733.4,
        "7.399.999.999.999.990": 7.4,
        "7.459.900.000.000.000": 745.99,
        "7.791.000.000.000.000": 77.91,
        "8.094.999.999.999.990": 80.95,
        "8.153.999.999.999.990": 81.54,
        "8.185.011.000.000.000": 818.5011,
        "8.319.000.000.000.000": 83.19,
    }
    
    
    df['Superficie_Total_Real'] = df['Superficie_Total_Real'].astype(str).str.strip().replace(MAPEO_SUPERFICIES_CORRUPTAS)
    
    
    cols_numericas = ['lat', 'lon', 'Superficie_Total_Real']
    for col in cols_numericas:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
    
    df_unicos = df.drop_duplicates(subset=['lat', 'lon', 'fecha_ini', 'Superficie_Total_Real']).copy()
    
    
    df_unicos['fecha_dt'] = pd.to_datetime(df_unicos['fecha_ini'])
    df_unicos = df_unicos.sort_values('fecha_dt').reset_index(drop=True)
    df_unicos = df_unicos.drop(columns=['fecha_dt'])
    
    print(f"Incendios brutos: {len(df)} | Únicos reales (Filtro Aplicado): {len(df_unicos)}")
    return df_unicos

# ============================================================
# 2. GENERACIÓN ESPACIAL DE AUSENCIAS
# ============================================================

def categorizar_vegetacion(valor):
    """Homologa las coberturas de la COSCV 2024 a nuestras 6 clases de combustible."""
    if pd.isna(valor):
        return "Urbano/Otros"
    v = str(valor).lower()
    if 'conífera' in v or 'conifera' in v:
        return "Coníferas"
    elif 'frondosa' in v or 'ribera' in v or 'mixto' in v:
        return "Frondosas"
    elif 'matorral' in v or 'cortafuegos' in v:
        return "Matorral"
    elif 'pastizal' in v or 'pasto' in v:
        return "Pastizal"
    elif 'secano' in v or 'regadío' in v or 'regadio' in v or 'invernadero' in v or 'huerta' in v or 'arable' in v or 'olivar' in v or 'frutales' in v or 'viñedo' in v or 'cítricos' in v:
        return "Agrícola"
    return "Urbano/Otros"

def generar_coordenadas_negativas(n_muestras):
    print(f"Cargando cartografía forestal COSCV 2024 usando pyogrio...")
    gdf_veg = gpd.read_file(GPKG_VEGETACION, engine='pyogrio')
    
    
    if gdf_veg.crs is None or gdf_veg.crs.to_string() != "EPSG:4326":
        print("Reproyectando mapa forestal a EPSG:4326...")
        gdf_veg = gdf_veg.to_crs("EPSG:4326")
        
    print("Filtrando máscara de vegetación forestal y rural...")
    gdf_veg['tipo_vegetacion'] = gdf_veg['clase'].apply(categorizar_vegetacion)
    
    
    minx, miny, maxx, maxy = gdf_forestal.total_bounds
    puntos_validos = []
    intentos = 0
    
    print("Generando coordenadas y validando intersección forestal...")
    pbar = tqdm(total=n_muestras, desc="Ausencias Espaciales")
    
    while len(puntos_validos) < n_muestras:
        lote_size = int((n_muestras - len(puntos_validos)) * 3.0)
        lote_size = max(lote_size, 100)
        
        lats_rand = np.random.uniform(miny, maxy, lote_size)
        lons_rand = np.random.uniform(minx, maxx, lote_size)
        
        pts = [Point(x, y) for x, y in zip(lons_rand, lats_rand)]
        gdf_pts = gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326")
        
        
        cruce = gpd.sjoin(gdf_pts, gdf_forestal[['tipo_vegetacion', 'geometry']], how='inner', predicate='intersects')
        
        for _, row in cruce.iterrows():
            if len(puntos_validos) < n_muestras:
                puntos_validos.append({
                    'lat': row.geometry.y,
                    'lon': row.geometry.x,
                    'tipo_vegetacion': row['tipo_vegetacion']
                })
                pbar.update(1)
                
        intentos += lote_size
        
    pbar.close()
    print(f"Muestreo espacial concluido. Intentos Monte Carlo: {intentos}")
    return pd.DataFrame(puntos_validos)

# ============================================================
# 3. ENRIQUECIMIENTO CLIMÁTICO
# ============================================================

def inyectar_clima_era5(df_neg, df_pos):
    """
    Inyecta el espejo climatológico a los negativos
    """
    
    print("Sincronizando temporalmente los puntos negativos con las fechas reales...")
    fechas_reales = df_pos['fecha_ini'].values
    df_neg['fecha_ini'] = np.random.choice(fechas_reales, size=len(df_neg), replace=True)
    
    
    cols_clima = ['temp_max', 'temp_media', 'temp_min', 'precipitacion', 
                  'viento_medio', 'racha_max', 'dir_viento', 'humedad_media', 
                  'prec_acum_7d', 'tmax_max_7d', 'dias_sin_lluvia']
    for c in cols_clima:
        df_neg[c] = np.nan
        
    
    df_neg['fecha_dia'] = pd.to_datetime(df_neg['fecha_ini']).dt.strftime('%Y-%m-%d')
    fechas_unicas = df_neg['fecha_dia'].unique()
    print(f"Total días únicos a consultar en ERA5: {len(fechas_unicas)}")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    
    for fecha_str in tqdm(fechas_unicas, desc="Consultando ERA5"):
        
        indices = df_neg[df_neg['fecha_dia'] == fecha_str].index.tolist()
        df_fecha = df_neg.loc[indices]
        
        
        fecha_end = pd.to_datetime(fecha_str)
        fecha_start = fecha_end - pd.Timedelta(days=7)
        
        start_str = fecha_start.strftime('%Y-%m-%d')
        end_str = fecha_end.strftime('%Y-%m-%d')
        
        
        chunk_size = 50
        for i in range(0, len(df_fecha), chunk_size):
            chunk = df_fecha.iloc[i:i+chunk_size]
            chunk_indices = indices[i:i+chunk_size]
            
            lats = chunk['lat'].tolist()
            lons = chunk['lon'].tolist()
            
            params = {
                "latitude": ",".join(map(str, lats)),
                "longitude": ",".join(map(str, lons)),
                "start_date": start_str,
                "end_date": end_str,
                "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant",
                "hourly": "relative_humidity_2m",
                "timezone": "Europe/Madrid"
            }
            
            
            daily_results = None
            hourly_results = None
            for intento in range(3):
                try:
                    res = requests.get(url, params=params, timeout=15)
                    if res.status_code == 200:
                        data = res.json()
                        results_list = data if isinstance(data, list) else [data]
                        daily_results = [r.get('daily', {}) for r in results_list]
                        hourly_results = [r.get('hourly', {}) for r in results_list]
                        break
                    elif res.status_code == 429:
                        time.sleep(2 * (intento + 1))
                except Exception:
                    time.sleep(1)
            
            if not daily_results:
                continue
                
            
            for j, idx in enumerate(chunk_indices):
                try:
                    daily = daily_results[j]
                    hourly = hourly_results[j]
                    
                    if not daily or not hourly:
                        continue
                        
                    
                    df_neg.at[idx, 'temp_media'] = daily['temperature_2m_mean'][-1]
                    df_neg.at[idx, 'temp_max'] = daily['temperature_2m_max'][-1]
                    df_neg.at[idx, 'temp_min'] = daily['temperature_2m_min'][-1]
                    df_neg.at[idx, 'precipitacion'] = daily['precipitation_sum'][-1]
                    df_neg.at[idx, 'viento_medio'] = daily['wind_speed_10m_max'][-1]
                    df_neg.at[idx, 'racha_max'] = daily['wind_gusts_10m_max'][-1]
                    df_neg.at[idx, 'dir_viento'] = daily['wind_direction_10m_dominant'][-1]
                    
                    
                    hum_media = np.mean(hourly['relative_humidity_2m'][-24:])
                    df_neg.at[idx, 'humedad_media'] = np.round(hum_media, 1)
                    
                    
                    prec_7d = daily['precipitation_sum'][:-1]
                    tmax_7d = daily['temperature_2m_max'][:-1]
                    
                    df_neg.at[idx, 'prec_acum_7d'] = sum([p for p in prec_7d if p is not None])
                    
                    valid_tmax = [t for t in tmax_7d if t is not None]
                    df_neg.at[idx, 'tmax_max_7d'] = max(valid_tmax) if valid_tmax else np.nan
                    
                    
                    dias_sin_lluvia = 0
                    for p in reversed(prec_7d):
                        if p is not None and p < 0.1:
                            dias_sin_lluvia += 1
                        else:
                            break
                    df_neg.at[idx, 'dias_sin_lluvia'] = dias_sin_lluvia
                    
                except Exception:
                    pass
            
            time.sleep(PAUSE_OPENMETEO)
            
    df_neg = df_neg.drop(columns=['fecha_dia'])
    return df_neg

# ============================================================
# 4. ENRIQUECIMIENTO TOPOGRÁFICO
# ============================================================

def get_elevations_batch(points):
    url = "https://api.opentopodata.org/v1/srtm30m"
    locations = "|".join([f"{lat},{lon}" for lat, lon in points])
    for intento in range(3):
        try:
            response = requests.get(url, params={"locations": locations}, timeout=15)
            if response.status_code == 200:
                return [r['elevation'] for r in response.json()['results']]
            elif response.status_code == 429:
                time.sleep(2 * (intento + 1))
        except Exception:
            time.sleep(1)
    return [np.nan] * len(points)

def inyectar_topografia_final(df_neg):
    print("Calculando coordenadas de matriz y extrayendo elevación de Open Topo Data...")
    delta = 0.0005  
    points_to_query = []
    
    
    for _, row in df_neg.iterrows():
        lat, lon = row['lat'], row['lon']
        points_to_query.append((lat, lon))               
        points_to_query.append((lat + delta, lon))       
        points_to_query.append((lat - delta, lon))       
        points_to_query.append((lat, lon + delta))       
        points_to_query.append((lat, lon - delta))       
        
    total_points = len(points_to_query)
    all_elevs = []
    
    
    batch_size = 100
    for i in tqdm(range(0, total_points, batch_size), desc="Topografía SRTM 30m"):
        batch = points_to_query[i:i+batch_size]
        elevs = get_elevations_batch(batch)
        all_elevs.extend(elevs)
        time.sleep(PAUSE_OPENTOPODATA)
        
    
    print("Efectuando cálculos vectoriales de pendiente y aspecto...")
    elevaciones = []
    pendientes = []
    orientaciones = []
    
    for i in range(len(df_neg)):
        base_idx = i * 5
        z_c = all_elevs[base_idx]
        z_n = all_elevs[base_idx + 1]
        z_s = all_elevs[base_idx + 2]
        z_e = all_elevs[base_idx + 3]
        z_w = all_elevs[base_idx + 4]
        
        lat = df_neg.iloc[i]['lat']
        
        
        dy = 2 * delta * 111320
        dx = 2 * delta * 111320 * np.cos(np.radians(lat))
        
        
        dz_dx = (z_e - z_w) / dx if not np.isnan(z_e) and not np.isnan(z_w) else np.nan
        dz_dy = (z_n - z_s) / dy if not np.isnan(z_n) and not np.isnan(z_s) else np.nan
        
        if np.isnan(dz_dx) or np.isnan(dz_dy):
            pendientes.append(np.nan)
            orientaciones.append(np.nan)
            elevaciones.append(z_c)
            continue
            
    
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)
        
        
        aspect_deg = np.degrees(np.arctan2(-dz_dx, -dz_dy))
        aspect_deg = (aspect_deg + 360) % 360
        
        if slope_deg < 0.1:
            aspect_deg = -1  
            
        pendientes.append(np.round(slope_deg, 2))
        orientaciones.append(np.round(aspect_deg, 2))
        elevaciones.append(np.round(z_c, 1) if not np.isnan(z_c) else np.nan)
        
    df_neg['elevacion'] = elevaciones
    df_neg['pendiente'] = pendientes
    df_neg['orientacion'] = orientaciones
    return df_neg

# ============================================================
# 5. UNIFICACIÓN Y EXPORTACIÓN DEL DATASET MODELADO
# ============================================================

def main():
    print("=============================================================")
    print(" INICIANDO PIPELINE DE GENERACIÓN DE PSEUDO-AUSENCIAS (3.1)")
    print("=============================================================")
    
    
    df_pos = cargar_incendios_limpios()
    n_ausencias = len(df_pos)  # Ratio 1:1
    
   
    if CSV_SALIDA.exists():
        print(f"Cargando ausencias pre-generadas desde {CSV_SALIDA.name} para optimizar...")
        df_existente = pd.read_csv(CSV_SALIDA, sep=';', decimal=',')
        df_neg = df_existente[df_existente['target'] == 0].copy()
        df_neg = df_neg.drop(columns=['target', 'Superficie_Total_Real'])
    else:
        df_neg = generar_coordenadas_negativas(n_ausencias)
        
       
        df_neg = inyectar_clima_era5(df_neg, df_pos)
        
       
        df_neg = inyectar_topografia_final(df_neg)
    
 
    cols_imputar = ['temp_max', 'temp_media', 'temp_min', 'precipitacion', 
                    'viento_medio', 'racha_max', 'humedad_media', 'prec_acum_7d', 
                    'tmax_max_7d', 'dias_sin_lluvia', 'elevacion', 'pendiente', 'orientacion']
    
    for c in cols_imputar:
        if df_neg[c].isna().any():
            mediana_pos = df_pos[c].median()
            df_neg[c] = df_neg[c].fillna(mediana_pos)
            
    
    df_neg['dias_sin_lluvia'] = df_neg['dias_sin_lluvia'].astype(int)
    
    
    df_pos['target'] = 1
    df_neg['target'] = 0
    
  
    df_pos = df_pos[df_neg.columns.tolist() + ['Superficie_Total_Real']]
    df_neg['Superficie_Total_Real'] = 0.0 
    
    dataset_final = pd.concat([df_pos, df_neg], ignore_index=True)
    
    
    columnas_orden = [
        'lat', 'lon', 'fecha_ini', 'target', 'Superficie_Total_Real',
        'tipo_vegetacion', 'elevacion', 'pendiente', 'orientacion',
        'temp_max', 'temp_media', 'temp_min', 'precipitacion',
        'viento_medio', 'racha_max', 'dir_viento', 'humedad_media',
        'prec_acum_7d', 'tmax_max_7d', 'dias_sin_lluvia'
    ]
    dataset_final = dataset_final[columnas_orden]
    
    print(f"\nConsolidando dataset unificado y balanceado...")
    print(f"Total registros: {len(dataset_final)} (Positivos: {len(df_pos)} | Negativos: {len(df_neg)})")
    
    print(f"Guardando dataset en: {CSV_SALIDA.name}...")
    dataset_final.to_csv(CSV_SALIDA, sep=';', index=False, decimal=',')
    print("¡Pipeline 3.1 ejecutado con éxito! Muestras negativas inyectadas y balanceadas.")
    print("=============================================================")

if __name__ == '__main__':
    main()
