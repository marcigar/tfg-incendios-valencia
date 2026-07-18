"""
Script para la generación de pseudo-ausencias aleatorias (Alternativa B - Muestreo Espacio-Temporal Puro).
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
import random
from pathlib import Path
from shapely.geometry import Point
from tqdm import tqdm


np.random.seed(42)
random.seed(42)


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATOS_PROCESSED = BASE_DIR / "datos" / "processed"
DATOS_EXTERNAL = BASE_DIR / "datos" / "external" / "vegetacion_cv"

GPKG_VEGETACION = DATOS_EXTERNAL / "04_2024COSCV_25830_GPKG" / "COScv2024.gpkg"
CSV_SALIDA = DATOS_PROCESSED / "01_coordenadas_fechas_ausencias_altB.csv"

N_MUESTRAS = 4476
INICIO_ANIO = 2010
FIN_ANIO = 2022

def categorizar_vegetacion(valor):
    """Homologa las coberturas de la COSCV a nuestras clases de combustible."""
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
    print(f"Cargando cartografía forestal COSCV 2024 usando pyogrio (optimizando columnas)...")
    if not GPKG_VEGETACION.exists():
        raise FileNotFoundError(f"No se encontró el archivo cartográfico en: {GPKG_VEGETACION}")
    
    
    gdf_veg = gpd.read_file(GPKG_VEGETACION, engine='pyogrio', columns=['clase', 'geometry'])
    
    
    if gdf_veg.crs is None or gdf_veg.crs.to_string() != "EPSG:4326":
        print("Reproyectando mapa forestal a EPSG:4326...")
        gdf_veg = gdf_veg.to_crs("EPSG:4326")
        
    print("Filtrando máscara de vegetación forestal y rural...")
    gdf_veg['tipo_vegetacion'] = gdf_veg['clase'].apply(categorizar_vegetacion)
    
    
    gdf_forestal = gdf_veg[gdf_veg['tipo_vegetacion'] != 'Urbano/Otros'].copy()
    
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

def generar_fechas_aleatorias(n, inicio_anio=2012, fin_anio=2024):
    print(f"Generando fechas y horas aleatorias entre {inicio_anio} y {fin_anio}...")
    start = pd.Timestamp(f"{inicio_anio}-01-01 00:00:00")
    end = pd.Timestamp(f"{fin_anio}-12-31 23:59:59")
    start_u = int(start.value // 10**9)
    end_u = int(end.value // 10**9)
    
    timestamps_rand = np.random.randint(start_u, end_u, size=n)
    fechas = pd.to_datetime(timestamps_rand, unit='s')
    return fechas.strftime('%Y-%m-%d %H:%M:%S')

def main():
    print("=============================================================")
    print(" INICIANDO PIPELINE DE GENERACIÓN DE PSEUDO-AUSENCIAS (ALT B)")
    print("=============================================================")
    DATOS_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    df_neg = generar_coordenadas_negativas(N_MUESTRAS)
    
    
    df_neg['fecha_ini'] = generar_fechas_aleatorias(N_MUESTRAS, INICIO_ANIO, FIN_ANIO)
    
    print(f"Guardando {len(df_neg)} ausencias espacio-temporales en: {CSV_SALIDA}")
    df_neg.to_csv(CSV_SALIDA, sep=';', index=False, decimal=',')
    print("Fase 1 completada con éxito")
    print("=============================================================")

if __name__ == '__main__':
    main()
