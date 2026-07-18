"""
Script para la extracción meteorológica  de pseudo-ausencias (Alternativa B - Muestreo Espacio-Temporal Puro).
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import sys
import io
from pathlib import Path
from tqdm import tqdm


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATOS_PROCESSED = BASE_DIR / "datos" / "processed"

CSV_INPUT = DATOS_PROCESSED / "01_coordenadas_fechas_ausencias_altB.csv"
CSV_OUTPUT = DATOS_PROCESSED / "02_ausencias_climatizadas_altB.csv"

URL_API = "https://archive-api.open-meteo.com/v1/archive"
CHUNK_SIZE = 10  
PAUSE_OPENMETEO = 0.1  


COLS_CLIMA = [
    'temp_max', 'temp_media', 'temp_min', 'precipitacion', 
    'viento_medio', 'racha_max', 'dir_viento', 'humedad_media', 
    'prec_acum_7d', 'tmax_max_7d', 'dias_sin_lluvia'
]

def obtener_clima_punto(lat, lon, fecha_ini_str):
    """Consulta la API de Open-Meteo para una coordenada y fecha."""
    fecha_end = pd.to_datetime(fecha_ini_str)
    fecha_start = fecha_end - pd.Timedelta(days=7)
    
    start_str = fecha_start.strftime('%Y-%m-%d')
    end_str = fecha_end.strftime('%Y-%m-%d')
    
    params = {
        "latitude": str(lat),
        "longitude": str(lon),
        "start_date": start_str,
        "end_date": end_str,
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant",
        "hourly": "relative_humidity_2m",
        "timezone": "Europe/Madrid"
    }
    
    for intento in range(5):
        try:
            r = requests.get(URL_API, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
                daily = data.get('daily', {})
                hourly = data.get('hourly', {})
                if daily and hourly:
                    return daily, hourly
            elif r.status_code == 429:
                time.sleep(3 * (intento + 1))
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1)
            
    return None, None

def procesar_registro(row):
    lat = row['lat']
    lon = row['lon']
    fecha_ini = row['fecha_ini']
    
    daily, hourly = obtener_clima_punto(lat, lon, fecha_ini)
    if daily is None or hourly is None:
        return None
        
    try:
        resultado = {
            'lat': lat,
            'lon': lon,
            'tipo_vegetacion': row['tipo_vegetacion'],
            'fecha_ini': fecha_ini,
            'temp_media': daily['temperature_2m_mean'][-1],
            'temp_max': daily['temperature_2m_max'][-1],
            'temp_min': daily['temperature_2m_min'][-1],
            'precipitacion': daily['precipitation_sum'][-1],
            'viento_medio': daily['wind_speed_10m_max'][-1],
            'racha_max': daily['wind_gusts_10m_max'][-1],
            'dir_viento': daily['wind_direction_10m_dominant'][-1],
        }
        
       
        hum_media = np.mean(hourly['relative_humidity_2m'][-24:])
        resultado['humedad_media'] = np.round(hum_media, 1)
        
        
        prec_7d = daily['precipitation_sum'][:-1]
        tmax_7d = daily['temperature_2m_max'][:-1]
        
        resultado['prec_acum_7d'] = sum([p for p in prec_7d if p is not None])
        
        valid_tmax = [t for t in tmax_7d if t is not None]
        resultado['tmax_max_7d'] = max(valid_tmax) if valid_tmax else np.nan
        
        
        dias_sin_lluvia = 0
        for p in reversed(prec_7d):
            if p is not None and p < 0.1:
                dias_sin_lluvia += 1
            else:
                break
        resultado['dias_sin_lluvia'] = int(dias_sin_lluvia)
        
        return resultado
    except Exception as e:
        print(f"Error procesando datos para {lat}, {lon} en {fecha_ini}: {e}")
        return None

def main():
    print("=============================================================")
    print(" INICIANDO PIPELINE DE ENRIQUECIMIENTO METEOROLÓGICO (ALT B) ")
    print("=============================================================")
    
    if not CSV_INPUT.exists():
        raise FileNotFoundError(f"No se encontró el archivo de ausencias: {CSV_INPUT}")
        
    df_input = pd.read_csv(CSV_INPUT, sep=';', decimal=',')
    total_registros = len(df_input)
    
    ya_procesados = 0
    if CSV_OUTPUT.exists():
        try:
            df_output_existente = pd.read_csv(CSV_OUTPUT, sep=';', decimal=',')
            ya_procesados = len(df_output_existente)
            print(f"Detectado archivo de salida parcial. Registros ya procesados: {ya_procesados}/{total_registros}")
            if ya_procesados >= total_registros:
                print("¡Todos los registros ya han sido procesados! Saliendo.")
                print("=============================================================")
                return
        except Exception as e:
            print(f"Error al leer archivo parcial existente (se reescribirá): {e}")
            ya_procesados = 0
            
    
    df_restante = df_input.iloc[ya_procesados:].reset_index(drop=True)
    n_restante = len(df_restante)
    print(f"Registros a descargar e integrar: {n_restante}")
    
    chunk_data = []
    
    
    pbar = tqdm(total=total_registros, initial=ya_procesados, desc="Descargando ERA5 (Alt B)")
    
    for idx, row in df_restante.iterrows():
        res = procesar_registro(row)
        
        
        if res is None:
            res = {
                'lat': row['lat'], 'lon': row['lon'], 
                'tipo_vegetacion': row['tipo_vegetacion'], 'fecha_ini': row['fecha_ini']
            }
            for col in COLS_CLIMA:
                res[col] = np.nan
        
        chunk_data.append(res)
        pbar.update(1)
        
        
        if len(chunk_data) >= CHUNK_SIZE or idx == n_restante - 1:
            df_chunk = pd.DataFrame(chunk_data)
            
            
            if not CSV_OUTPUT.exists() or (ya_procesados == 0 and idx < CHUNK_SIZE):
                df_chunk.to_csv(CSV_OUTPUT, sep=';', index=False, decimal=',', encoding='utf-8')
            else:
                
                df_chunk.to_csv(CSV_OUTPUT, sep=';', mode='a', header=False, index=False, decimal=',', encoding='utf-8')
                
            chunk_data = []  #
            
        time.sleep(PAUSE_OPENMETEO)
        
    pbar.close()
    print(f"Pipeline de clima completado. Resultados guardados en: {CSV_OUTPUT}")
    print("=============================================================")

if __name__ == '__main__':
    main()
