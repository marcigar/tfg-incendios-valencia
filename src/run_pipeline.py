"""
Script de generacion del dataset unificado.
"""
import sys, os, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import requests
import time
import json
from sklearn.neighbors import BallTree
from tqdm import tqdm

from descarga_aemet import (
    cargar_api_key,
    obtener_estaciones_cv,
    limpiar_datos_clima,
    COLUMNAS_CLIMA,
)

BASE = os.path.join(os.path.dirname(__file__), '..')
RUTA_INCENDIOS = os.path.join(BASE, 'datos', 'processed', '01_incendios_CV_limpio.csv')
RUTA_CACHE = os.path.join(BASE, 'datos', 'external', 'cache_aemet')
RUTA_SALIDA = os.path.join(BASE, 'datos', 'processed', '03_dataset_final_CV.csv')



def paso1_cargar_clima_cache():
    print("=" * 60)
    print("  PASO 1: CARGAR DATOS CLIMÁTICOS DESDE CACHE")
    print("=" * 60)

    todos = []
    estaciones_con_datos = []

    for f in sorted(os.listdir(RUTA_CACHE)):
        ruta = os.path.join(RUTA_CACHE, f)
        if os.path.getsize(ruta) <= 10:
            continue
        try:
            with open(ruta, 'r', encoding='utf-8') as fh:
                datos = json.load(fh)
            if datos:
                todos.extend(datos)
                estaciones_con_datos.append(f.replace('.json', ''))
        except Exception:
            pass

    print(f"   Estaciones con datos: {len(estaciones_con_datos)}")
    print(f"   Registros diarios totales: {len(todos)}")

    df_clima_raw = pd.DataFrame(todos)
    df_clima = limpiar_datos_clima(df_clima_raw)

    return df_clima, set(estaciones_con_datos)



def paso2_asignar_con_fallback(df_inc, df_est, estaciones_validas):
    print("\n" + "=" * 60)
    print("  PASO 2: ASIGNAR ESTACIÓN CON FALLBACK")
    print("=" * 60)

    
    est_rad = np.deg2rad(df_est[['lat_est', 'lon_est']].values)
    inc_rad = np.deg2rad(df_inc[['lat', 'lon']].values)

    
    tree = BallTree(est_rad, metric='haversine')
    distancias, indices = tree.query(inc_rad, k=5)

    
    distancias_km = distancias * 6371

    
    id_estacion = []
    nombre_estacion = []
    dist_estacion = []
    rango_fallback = []

    for i in range(len(df_inc)):
        asignada = False
        for j in range(5):  
            idx = indices[i, j]
            est_id = df_est.iloc[idx]['indicativo']
            if est_id in estaciones_validas:
                id_estacion.append(est_id)
                nombre_estacion.append(df_est.iloc[idx]['nombre'])
                dist_estacion.append(round(distancias_km[i, j], 2))
                rango_fallback.append(j + 1)  
                asignada = True
                break
        if not asignada:
            
            idx = indices[i, 0]
            id_estacion.append(df_est.iloc[idx]['indicativo'])
            nombre_estacion.append(df_est.iloc[idx]['nombre'])
            dist_estacion.append(round(distancias_km[i, 0], 2))
            rango_fallback.append(0) 

    df_res = df_inc.copy()
    df_res['id_estacion'] = id_estacion
    df_res['nombre_estacion'] = nombre_estacion
    df_res['distancia_estacion_km'] = dist_estacion
    df_res['fallback_orden'] = rango_fallback

    
    con_datos = sum(1 for r in rango_fallback if r > 0)
    por_primera = sum(1 for r in rango_fallback if r == 1)
    por_fallback = sum(1 for r in rango_fallback if r > 1)
    sin_datos = sum(1 for r in rango_fallback if r == 0)

    print(f"   Asignados con 1a estacion: {por_primera}")
    print(f"   Asignados por fallback:    {por_fallback}")
    print(f"   Sin datos disponibles:     {sin_datos}")
    print(f"   Distancia media:           {np.mean(dist_estacion):.1f} km")
    print(f"   Distancia P95:             {np.percentile(dist_estacion, 95):.1f} km")

    return df_res



def paso3_cruzar(df_inc, df_clima):
    print("\n" + "=" * 60)
    print("  PASO 3: CRUZAR INCENDIOS CON CLIMA")
    print("=" * 60)

    df_i = df_inc.copy()
    df_i['fecha_ini'] = pd.to_datetime(df_i['fecha_ini'])
    df_i['fecha_cruce'] = df_i['fecha_ini'].dt.normalize()

    df_c = df_clima.copy()
    df_c['fecha_cruce'] = pd.to_datetime(df_c['fecha_clima']).dt.normalize()

    df_merged = df_i.merge(
        df_c, left_on=['id_estacion', 'fecha_cruce'],
        right_on=['id_estacion', 'fecha_cruce'], how='left'
    )
    df_merged = df_merged.drop(columns=['fecha_cruce', 'fecha_clima'], errors='ignore')

    total = len(df_merged)
    con_clima = df_merged['temp_media'].notna().sum()
    print(f"   Incendios con datos clima: {con_clima}/{total} ({con_clima/total*100:.1f}%)")

    return df_merged



def paso4_variables_previas(df_inc, df_clima, dias=7):
    print(f"\n" + "=" * 60)
    print(f"  PASO 4: VARIABLES DE LOS {dias} DÍAS PREVIOS")
    print("=" * 60)

    df_cl = df_clima.copy()
    df_cl['fecha_clima'] = pd.to_datetime(df_cl['fecha_clima'])
    df_cl = df_cl.sort_values(['id_estacion', 'fecha_clima'])

    frames = []
    for est, g in df_cl.groupby('id_estacion'):
        g = g.set_index('fecha_clima').sort_index()
        idx = pd.date_range(g.index.min(), g.index.max(), freq='D')
        g = g.reindex(idx)
        g['id_estacion'] = est

        if 'precipitacion' in g.columns:
            g[f'prec_acum_{dias}d'] = (
                g['precipitacion'].rolling(window=dias, min_periods=1).sum().shift(1)
            )
            llueve = (g['precipitacion'].fillna(0) > 0.1).astype(int)
            bloques = llueve.cumsum()
            g['dias_sin_lluvia'] = g.groupby(bloques).cumcount()
            g['dias_sin_lluvia'] = g['dias_sin_lluvia'].shift(1)

        if 'temp_max' in g.columns:
            g[f'tmax_max_{dias}d'] = (
                g['temp_max'].rolling(window=dias, min_periods=1).max().shift(1)
            )

        g['fecha_cruce'] = g.index
        frames.append(g.reset_index(drop=True))

    df_rolling = pd.concat(frames, ignore_index=True)

    cols = ['id_estacion', 'fecha_cruce',
            f'prec_acum_{dias}d', 'dias_sin_lluvia', f'tmax_max_{dias}d']
    cols = [c for c in cols if c in df_rolling.columns]
    df_r = df_rolling[cols].copy()

    df_i = df_inc.copy()
    df_i['fecha_ini'] = pd.to_datetime(df_i['fecha_ini'])
    df_i['fecha_cruce'] = df_i['fecha_ini'].dt.normalize()

    df_res = df_i.merge(df_r, on=['id_estacion', 'fecha_cruce'], how='left')
    df_res = df_res.drop(columns=['fecha_cruce'], errors='ignore')

    print(f"   Variables previas calculadas")
    return df_res



ELEV_API = "https://api.open-meteo.com/v1/elevation"

def paso5_elevacion(df):
    print("\n" + "=" * 60)
    print("  PASO 5: ELEVACIÓN DEL TERRENO")
    print("=" * 60)

    df['_lat_r'] = df['lat'].round(4)
    df['_lon_r'] = df['lon'].round(4)
    coords = df[['_lat_r', '_lon_r']].drop_duplicates().reset_index(drop=True)
    total = len(coords)
    print(f"   Coordenadas únicas: {total}")

    elevs_all = []
    CHUNK = 80

    for i in tqdm(range(0, total, CHUNK), desc="Elevación"):
        c = coords.iloc[i:i+CHUNK]
        lats = ",".join(str(x) for x in c['_lat_r'])
        lons = ",".join(str(x) for x in c['_lon_r'])

        ok = False
        for intento in range(5):
            try:
                r = requests.get(ELEV_API, params={"latitude": lats, "longitude": lons}, timeout=20)
                if r.status_code == 200 and "elevation" in r.json():
                    elevs_all.extend(r.json()["elevation"])
                    ok = True
                    break
                elif r.status_code == 429:
                    time.sleep(3 * (intento + 1))
                    continue
            except Exception as e:
                time.sleep(2)
        if not ok:
            elevs_all.extend([np.nan] * len(c))
        time.sleep(0.3)

    coords['elevacion'] = elevs_all
    df = df.merge(coords, on=['_lat_r', '_lon_r'], how='left')
    df = df.drop(columns=['_lat_r', '_lon_r'])

    nulos = df['elevacion'].isna().sum()
    print(f"   Elevación OK. NaN: {nulos}/{len(df)}")
    return df



def paso6_guardar(df):
    print("\n" + "=" * 60)
    print("  PASO 6: DATASET FINAL")
    print("=" * 60)

    cols_orden = [
        'Campania', 'fecha_ini', 'duracion_h',
        'Provincia', 'Municipio', 'Causa',
        'lat', 'lon', 'elevacion',
        'Superficie_Total_Real', 'Superficie_Arbolada',
        'AfectoZonasInterfazUrbanoForestal', 'AfectoEspacioProtegido',
        'id_estacion', 'nombre_estacion', 'distancia_estacion_km',
        'temp_media', 'temp_max', 'temp_min',
        'precipitacion', 'humedad_media',
        'viento_medio', 'racha_max', 'dir_viento',
        'prec_acum_7d', 'dias_sin_lluvia', 'tmax_max_7d',
    ]
    cols_final = [c for c in cols_orden if c in df.columns]
    for c in df.columns:
        if c not in cols_final and c != 'fallback_orden':
            cols_final.append(c)

    df = df[cols_final]
    df.to_csv(RUTA_SALIDA, sep=';', index=False, encoding='utf-8-sig')

    print(f"\n   Archivo: {os.path.basename(RUTA_SALIDA)}")
    print(f"   Registros: {len(df)}")
    print(f"   Columnas:  {len(df.columns)}")
    print(f"\n   Completitud:")
    for col in df.columns:
        pct = (1 - df[col].isna().mean()) * 100
        print(f"     {col}: {pct:.0f}%")

    return df



if __name__ == '__main__':
    print("=" * 60)
    print("  PIPELINE FINAL - TFG Incendios C. Valenciana")
    print("=" * 60)

    try:
        
        df_clima, est_validas = paso1_cargar_clima_cache()

        
        df_inc = pd.read_csv(RUTA_INCENDIOS, sep=';')
        df_inc['fecha_ini'] = pd.to_datetime(df_inc['fecha_ini'], dayfirst=True, format='mixed')
        df_inc['lat'] = pd.to_numeric(df_inc['lat'], errors='coerce')
        df_inc['lon'] = pd.to_numeric(df_inc['lon'], errors='coerce')
        df_inc = df_inc.dropna(subset=['lat', 'lon', 'fecha_ini'])
        print(f"\n   Incendios cargados: {len(df_inc)}")

        
        api_key = cargar_api_key()
        df_est = obtener_estaciones_cv(api_key)

    
        df_inc = paso2_asignar_con_fallback(df_inc, df_est, est_validas)

        
        df_merged = paso3_cruzar(df_inc, df_clima)

        
        df_merged = paso4_variables_previas(df_merged, df_clima, dias=7)

        
        df_merged = paso5_elevacion(df_merged)

        
        df_final = paso6_guardar(df_merged)

        print("\n" + "=" * 60)
        print("  PIPELINE COMPLETADO CON ÉXITO!")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
