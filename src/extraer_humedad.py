import pandas as pd
import requests
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
INPUT_CSV = os.path.join(BASE_DIR, 'datos', 'processed', '04b_dataset_elevacion_fixed.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'datos', 'processed', '05_dataset_final_para_EDA.csv')

def fetch_humidity(row):
    idx, lat, lon, fecha = row
    fecha_dt = pd.to_datetime(fecha).strftime('%Y-%m-%d')
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": fecha_dt,
        "end_date": fecha_dt,
        "hourly": "relative_humidity_2m",
        "timezone": "Europe/Madrid"
    }
    
    for _ in range(3):
        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if 'hourly' in data and 'relative_humidity_2m' in data['hourly']:
                    humedades = [h for h in data['hourly']['relative_humidity_2m'] if h is not None]
                    if humedades:
                        return idx, sum(humedades)/len(humedades)
            elif res.status_code == 429:
                time.sleep(2)
            else:
                time.sleep(0.5)
        except Exception:
            time.sleep(0.5)
            
    return idx, None

def main():
    print("Cargando dataset para procesar humedad...")
    df = pd.read_csv(INPUT_CSV, sep=';')
    
   
    df['lat_round'] = df['lat'].round(2)
    df['lon_round'] = df['lon'].round(2)
    df['fecha_solo'] = pd.to_datetime(df['fecha_ini']).dt.strftime('%Y-%m-%d')
    
    unique_requests = df[['lat_round', 'lon_round', 'fecha_solo']].drop_duplicates()
    print(f"Total registros: {len(df)}. Peticiones unicas optimizadas: {len(unique_requests)}")
    
    tareas = []
    for idx, row in unique_requests.iterrows():
        tareas.append((idx, row['lat_round'], row['lon_round'], row['fecha_solo']))
        
    resultados = {}
    
    print("Extrayendo humedad relativa de Open-Meteo (Async)...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_humidity, t): t for t in tareas}
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, humedad = future.result()
            if humedad is not None:
                resultados[idx] = humedad
            time.sleep(0.05) 
            
    
    unique_requests['humedad_obtenida'] = unique_requests.index.map(resultados)
    
    
    df = df.merge(unique_requests[['lat_round', 'lon_round', 'fecha_solo', 'humedad_obtenida']], 
                  on=['lat_round', 'lon_round', 'fecha_solo'], how='left')
    
    df['humedad_media'] = df['humedad_obtenida'].round(1)
    df = df.drop(columns=['lat_round', 'lon_round', 'fecha_solo', 'humedad_obtenida'])
    
    
    for col in ['racha_max', 'dir_viento', 'prec_acum_7d', 'tmax_max_7d', 'elevacion', 'humedad_media']:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
            
    print("\n--- REPORTE FINAL DE NULOS ---")
    print(df.isna().sum().to_string())
    
    df.to_csv(OUTPUT_CSV, sep=';', index=False)
    print(f"\nArchivo definitivo generado: {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
