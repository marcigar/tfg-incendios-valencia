"""
Script para la extracción topográfica (Open Topo Data) de pseudo-ausencias (Alternativa B - Muestreo Espacio-Temporal Puro).
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

CSV_INPUT = DATOS_PROCESSED / "02_ausencias_climatizadas_altB.csv"
CSV_OUTPUT = DATOS_PROCESSED / "03_ausencias_topografia_altB.csv"

URL_API = "https://api.opentopodata.org/v1/srtm30m"
PAUSE_OPENTOPODATA = 1.0  
BATCH_LOCATIONS = 20  

def get_elevations_batch(points):
    """Consulta la API de Open Topo Data para una lista de coordenadas."""
    locations = "|".join([f"{lat},{lon}" for lat, lon in points])
    for intento in range(5):
        try:
            response = requests.get(URL_API, params={"locations": locations}, timeout=15)
            if response.status_code == 200:
                results = response.json().get('results', [])
                return [r['elevation'] for r in results]
            elif response.status_code == 429:
                time.sleep(3 * (intento + 1))
        except Exception:
            time.sleep(1)
    return [np.nan] * len(points)

def main():
    print("=============================================================")
    print(" INICIANDO PIPELINE DE ENRIQUECIMIENTO TOPOGRÁFICO (ALT B)    ")
    print("=============================================================")
    
    if not CSV_INPUT.exists():
        raise FileNotFoundError(f"No se encontró el archivo de clima: {CSV_INPUT}")
        
    df_input = pd.read_csv(CSV_INPUT, sep=';', decimal=',')
    total_registros = len(df_input)
    
   
    ya_procesados = 0
    if CSV_OUTPUT.exists():
        try:
            df_output = pd.read_csv(CSV_OUTPUT, sep=';', decimal=',')
            ya_procesados = len(df_output)
            print(f"Detectado archivo de salida parcial. Registros ya procesados: {ya_procesados}/{total_registros}")
            if ya_procesados >= total_registros:
                print("¡Todos los registros ya cuentan con datos topográficos! Saliendo.")
                print("=============================================================")
                return
        except Exception as e:
            print(f"Error al leer archivo parcial (se reescribirá): {e}")
            ya_procesados = 0
            
    df_restante = df_input.iloc[ya_procesados:].reset_index(drop=True)
    n_restante = len(df_restante)
    print(f"Registros topográficos a extraer: {n_restante}")
    
    delta = 0.0005  
    
    
    for i in tqdm(range(0, n_restante, BATCH_LOCATIONS), desc="Topografía SRTM 30m"):
        lote = df_restante.iloc[i:i+BATCH_LOCATIONS]
        
        
        puntos_consulta = []
        for _, row in lote.iterrows():
            lat, lon = row['lat'], row['lon']
            puntos_consulta.append((lat, lon))               
            puntos_consulta.append((lat + delta, lon))       
            puntos_consulta.append((lat - delta, lon))       
            puntos_consulta.append((lat, lon + delta))      
            puntos_consulta.append((lat, lon - delta))       
            
        elevs = get_elevations_batch(puntos_consulta)
        
       
        resultados_lote = []
        for j, (_, row) in enumerate(lote.iterrows()):
            base_idx = j * 5
            z_c = elevs[base_idx]
            z_n = elevs[base_idx + 1]
            z_s = elevs[base_idx + 2]
            z_e = elevs[base_idx + 3]
            z_w = elevs[base_idx + 4]
            
            lat = row['lat']
            dy = 2 * delta * 111320
            dx = 2 * delta * 111320 * np.cos(np.radians(lat))
            
            
            dz_dx = (z_e - z_w) / dx if not np.isnan(z_e) and not np.isnan(z_w) else np.nan
            dz_dy = (z_n - z_s) / dy if not np.isnan(z_n) and not np.isnan(z_s) else np.nan
            
            if np.isnan(dz_dx) or np.isnan(dz_dy):
                pendiente = np.nan
                orientacion = np.nan
                elevacion = z_c
            else:
                slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
                aspect_deg = (np.degrees(np.arctan2(-dz_dx, -dz_dy)) + 360) % 360
                if slope_deg < 0.1:
                    aspect_deg = -1  # Terreno plano
                    
                pendiente = np.round(slope_deg, 2)
                orientacion = np.round(aspect_deg, 2)
                elevacion = np.round(z_c, 1) if not np.isnan(z_c) else np.nan
                
            
            res_row = row.to_dict()
            res_row['elevacion'] = elevacion
            res_row['pendiente'] = pendiente
            res_row['orientacion'] = orientacion
            
            resultados_lote.append(res_row)
            
        
        df_lote = pd.DataFrame(resultados_lote)
        
        
        cols_orden = list(df_input.columns) + ['elevacion', 'pendiente', 'orientacion']
        df_lote = df_lote[cols_orden]
        
        if not CSV_OUTPUT.exists() or (ya_procesados == 0 and i == 0):
            df_lote.to_csv(CSV_OUTPUT, sep=';', index=False, decimal=',', encoding='utf-8')
        else:
            df_lote.to_csv(CSV_OUTPUT, sep=';', mode='a', header=False, index=False, decimal=',', encoding='utf-8')
            
        time.sleep(PAUSE_OPENTOPODATA)
        
    print(f"Pipeline de topografía completado. Resultados en: {CSV_OUTPUT}")
    print("=============================================================")

if __name__ == '__main__':
    main()
