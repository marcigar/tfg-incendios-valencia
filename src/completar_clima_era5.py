"""
Script para completar datos climáticos faltantes usando ERA5 vía Open-Meteo.
TFG: Análisis y Predicción de Riesgo de Incendios en la C. Valenciana
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, timedelta
from tqdm import tqdm




BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
INPUT_CSV = os.path.join(BASE_DIR, 'datos', 'processed', '03_dataset_final_CV.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'datos', 'processed', '04_dataset_completo_clima_elevacion.csv')


API_PAUSE = 0.2


COLS_CHECK = ['viento_medio', 'temp_media', 'precipitacion']


def get_era5_data(lat, lon, fecha_ini):
    fecha_end = pd.to_datetime(fecha_ini)
    fecha_start = fecha_end - pd.Timedelta(days=7)
    
    start_str = fecha_start.strftime('%Y-%m-%d')
    end_str = fecha_end.strftime('%Y-%m-%d')
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_str,
        "end_date": end_str,
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant",
        "timezone": "Europe/Madrid"
    }
    
    max_retries = 3
    for intento in range(max_retries):
        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if 'daily' in data:
                    return data['daily']
            elif res.status_code == 429:
                time.sleep(2 * (intento + 1))
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1)
            
    return None

def extract_features(daily_data):
    """
    Extrae las variables del día del incendio y las variables de los 7 días previos.
    """
    try:
        
        temp_media = daily_data['temperature_2m_mean'][-1]
        temp_max = daily_data['temperature_2m_max'][-1]
        temp_min = daily_data['temperature_2m_min'][-1]
        precipitacion = daily_data['precipitation_sum'][-1]
        viento_medio = daily_data['wind_speed_10m_max'][-1]
        racha_max = daily_data['wind_gusts_10m_max'][-1]
        dir_viento = daily_data['wind_direction_10m_dominant'][-1]
        
        
        prec_7d = daily_data['precipitation_sum'][:-1]
        tmax_7d = daily_data['temperature_2m_max'][:-1]
        
        
        prec_acum_7d = sum([p for p in prec_7d if p is not None])
        
        
        valid_tmax = [t for t in tmax_7d if t is not None]
        tmax_max_7d = max(valid_tmax) if valid_tmax else None
        
        
        dias_sin_lluvia = 0
        for p in reversed(prec_7d):
            if p is not None and p < 0.1:
                dias_sin_lluvia += 1
            else:
                break
                
        return {
            'temp_media': temp_media,
            'temp_max': temp_max,
            'temp_min': temp_min,
            'precipitacion': precipitacion,
            'viento_medio': viento_medio,
            'racha_max': racha_max,
            'dir_viento': dir_viento,
            'prec_acum_7d': prec_acum_7d,
            'tmax_max_7d': tmax_max_7d,
            'dias_sin_lluvia': dias_sin_lluvia
        }
    except Exception as e:
        return None



def main(test_mode=True):
    print(f"Cargando dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, sep=';')
    
    mask_nulos = df[COLS_CHECK].isna().any(axis=1)
    idx_nulos = df[mask_nulos].index.tolist()
    
    print(f"   Total de incendios: {len(df)}")
    print(f"   Incendios con datos climáticos faltantes: {len(idx_nulos)}")
    
    if test_mode:
        print("\n[MODO PRUEBA] Procesando solo los primeros 10 registros nulos...")
        idx_nulos = idx_nulos[:10]
    else:
        print("\n[MODO PRODUCCION] Procesando todos los registros nulos...")
        
    exitos = 0
    fallos = 0
    
    
    for idx in tqdm(idx_nulos, desc="Consultando ERA5 (Open-Meteo)"):
        row = df.loc[idx]
        lat = row['lat']
        lon = row['lon']
        fecha_ini = row['fecha_ini']
        
        
        if pd.isna(lat) or pd.isna(lon) or pd.isna(fecha_ini):
            fallos += 1
            continue
            
        try:
            fecha_dt = pd.to_datetime(fecha_ini, dayfirst=True)
            
        except Exception:
            fallos += 1
            continue
            
        daily_data = get_era5_data(lat, lon, fecha_dt)
        
        if daily_data:
            features = extract_features(daily_data)
            if features:
    
                for col, val in features.items():
                    df.at[idx, col] = val
                exitos += 1
            else:
                fallos += 1
        else:
            fallos += 1
            
        time.sleep(API_PAUSE)
        
    print(f"\nProceso completado. Exitos: {exitos}, Fallos: {fallos}")
    
    if test_mode:
        print("\nMuestra de los resultados (primeros 5 imputados):")
        columnas_mostrar = ['fecha_ini', 'temp_media', 'precipitacion', 'viento_medio', 'prec_acum_7d', 'dias_sin_lluvia']
        print(df.loc[idx_nulos[:5], columnas_mostrar])
        print("\nNo se guarda el CSV en modo prueba.")
    else:
        print(f"\nGuardando dataset final en: {OUTPUT_CSV}")
        df.to_csv(OUTPUT_CSV, sep=';', index=False)
        print("¡Completado con éxito!")

if __name__ == "__main__":
    main(test_mode=False)
