import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path

def get_elevations_batch(points):
    """
    Obtiene elevaciones para una lista de puntos (lat, lon) usando Open Topo Data.
    """
    url = "https://api.opentopodata.org/v1/srtm30m"
    locations = "|".join([f"{lat},{lon}" for lat, lon in points])
    try:
        response = requests.get(url, params={"locations": locations})
        if response.status_code == 200:
            return [r['elevation'] for r in response.json()['results']]
        else:
            print(f"Error en API ({response.status_code}): {response.text}")
            return [np.nan] * len(points)
    except Exception as e:
        print(f"Excepción en API: {e}")
        return [np.nan] * len(points)

def main():
    BASE_DIR = Path("c:/Users/marco/Desktop/tfg")
    DATOS_PROCESSED = BASE_DIR / "datos" / "processed"
    
    csv_in = DATOS_PROCESSED / "06_dataset_final_enriquecido.csv"
    csv_out = DATOS_PROCESSED / "07_dataset_final_TOPOGRAFICO.csv"
    
    print(f"Leyendo {csv_in.name}...")
    df = pd.read_csv(csv_in, sep=';')
    
    
    if 'AfectoZonasInterfazUrbanoForestal' in df.columns:
        df = df.drop(columns=['AfectoZonasInterfazUrbanoForestal'])
        print("Columna 'AfectoZonasInterfazUrbanoForestal' eliminada.")
    
    
    delta = 0.0005  
    points_to_query = []
    
    for _, row in df.iterrows():
        lat, lon = row['lat'], row['lon']
        points_to_query.append((lat, lon))               
        points_to_query.append((lat + delta, lon))       
        points_to_query.append((lat - delta, lon))       
        points_to_query.append((lat, lon + delta))       
        points_to_query.append((lat, lon - delta))       
        
    
    batch_size_points = 100
    all_elevs = []
    
    total_points = len(points_to_query)
    print(f"Consultando elevaciones para {total_points} puntos (batches de 100)...")
    
    for i in range(0, total_points, batch_size_points):
        batch = points_to_query[i:i + batch_size_points]
        elevs = get_elevations_batch(batch)
        all_elevs.extend(elevs)
        
        if (i // batch_size_points) % 10 == 0:
            print(f"Progreso: {i}/{total_points} puntos...")
        
        
        time.sleep(1.0)
        
    
    print("Calculando topografía...")
    pendientes = []
    orientaciones = []
    
    for i in range(len(df)):
        base_idx = i * 5
        z_c = all_elevs[base_idx]
        z_n = all_elevs[base_idx + 1]
        z_s = all_elevs[base_idx + 2]
        z_e = all_elevs[base_idx + 3]
        z_w = all_elevs[base_idx + 4]
        
        lat = df.iloc[i]['lat']
        
        
        dy = 2 * delta * 111320
        dx = 2 * delta * 111320 * np.cos(np.radians(lat))
        
        
        dz_dx = (z_e - z_w) / dx
        dz_dy = (z_n - z_s) / dy
        
        
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)
        
        
        aspect_deg = np.degrees(np.arctan2(-dz_dx, -dz_dy))
        aspect_deg = (aspect_deg + 360) % 360
        
        
        if slope_deg < 0.1:
            aspect_deg = -1  
            
        pendientes.append(slope_deg)
        orientaciones.append(aspect_deg)
        
    df['pendiente'] = np.round(pendientes, 2)
    df['orientacion'] = np.round(orientaciones, 2)
    
    
    print(f"Guardando en {csv_out.name}...")
    df.to_csv(csv_out, sep=';', index=False)
    print("¡Hito conseguido! Dataset 07_TOPOGRAFICO listo.")

if __name__ == "__main__":
    main()
