"""
Módulo de descarga y cruce de datos meteorológicos de AEMET OpenData.
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import json
from datetime import datetime, timedelta
from sklearn.neighbors import BallTree
from tqdm import tqdm



CV_LAT_MIN, CV_LAT_MAX = 37.8, 40.85
CV_LON_MIN, CV_LON_MAX = -1.6, 0.6


CHUNK_DAYS = 150


API_PAUSE = 2.0


COLUMNAS_CLIMA = {
    'indicativo': 'id_estacion',
    'fecha': 'fecha_clima',
    'tmed': 'temp_media',
    'tmax': 'temp_max',
    'tmin': 'temp_min',
    'prec': 'precipitacion',
    'hrMedia': 'humedad_media',
    'velmedia': 'viento_medio',
    'racha': 'racha_max',
    'dir': 'dir_viento'
}


def cargar_api_key(ruta_api=None):
    """Carga la API key de AEMET desde archivo de texto."""
    if ruta_api is None:
        ruta_api = os.path.join(
            os.path.dirname(__file__), '..', 'datos', 'APIs', 'api_AEMET.txt'
        )
    with open(ruta_api, 'r') as f:
        return f.read().strip()




def consultar_aemet(endpoint, api_key, max_reintentos=3):
    url = f"https://opendata.aemet.es/opendata/api/{endpoint}"
    params = {"api_key": api_key}
    headers = {"cache-control": "no-cache"}

    for intento in range(max_reintentos):
        try:
            
            res = requests.get(url, params=params, headers=headers, timeout=30)

            if res.status_code == 429:
                wait = 60 * (intento + 1)
                print(f"   Rate limit alcanzado. Esperando {wait}s...")
                time.sleep(wait)
                continue

            if res.status_code != 200:
                print(f"   HTTP {res.status_code} en intento {intento+1}")
                time.sleep(API_PAUSE * 2)
                continue

            respuesta = res.json()
            datos_url = respuesta.get('datos')

            if not datos_url:
                desc = respuesta.get('descripcion', 'Sin descripción')
                if 'No hay datos' in desc:
                    return None
                print(f"   Sin URL de datos: {desc}")
                return None

           
            time.sleep(API_PAUSE)
            res_datos = requests.get(datos_url, timeout=60)

            if res_datos.status_code == 200:
                return res_datos.json()
            else:
                print(f"   Error descargando datos: HTTP {res_datos.status_code}")

        except requests.exceptions.Timeout:
            print(f"   Timeout en intento {intento+1}")
            time.sleep(API_PAUSE * 2)
        except json.JSONDecodeError:
            print(f"   Error JSON en intento {intento+1}")
            time.sleep(API_PAUSE)
        except Exception as e:
            print(f"   Error inesperado en intento {intento+1}: {e}")
            time.sleep(API_PAUSE)

    print(f"   Fallaron todos los reintentos para: {endpoint[:80]}...")
    return None



def _aemet_coord_to_decimal(coord, es_latitud=True):
    if pd.isna(coord) or coord == "":
        return None

    coord_str = str(coord).strip()
    direccion = coord_str[-1].upper()
    signo = -1 if direccion in ('W', 'S') else 1

    num_str = ''.join(filter(str.isdigit, coord_str))

    try:
        if es_latitud:
            num_str = num_str.zfill(6)
            grados = int(num_str[:2])
            minutos = int(num_str[2:4])
            segundos = int(num_str[4:6])
        else:
            num_str = num_str.zfill(7)
            grados = int(num_str[:3])
            minutos = int(num_str[3:5])
            segundos = int(num_str[5:7])
        return signo * (grados + minutos / 60 + segundos / 3600)
    except (ValueError, IndexError):
        return None


def obtener_estaciones_cv(api_key):
    """
    Descarga inventario de estaciones AEMET y filtra las de la C. Valenciana
    usando bounding box geográfico
    """
    print(" Descargando inventario de estaciones AEMET...")

    raw = consultar_aemet(
        "valores/climatologicos/inventarioestaciones/todasestaciones", api_key
    )
    if raw is None:
        raise RuntimeError("No se pudo descargar el inventario de estaciones")

    df = pd.DataFrame(raw)
    print(f"   Total estaciones España: {len(df)}")

    
    df['lat_est'] = df['latitud'].apply(
        lambda x: _aemet_coord_to_decimal(x, es_latitud=True)
    )
    df['lon_est'] = df['longitud'].apply(
        lambda x: _aemet_coord_to_decimal(x, es_latitud=False)
    )
    df = df.dropna(subset=['lat_est', 'lon_est'])

    
    margen = 0.15
    df_cv = df[
        (df['lat_est'] >= CV_LAT_MIN - margen) &
        (df['lat_est'] <= CV_LAT_MAX + margen) &
        (df['lon_est'] >= CV_LON_MIN - margen) &
        (df['lon_est'] <= CV_LON_MAX + margen)
    ].copy()

    print(f"   ✅{len(df_cv)} estaciones en la Comunidad Valenciana")
    return df_cv




def asignar_estacion_cercana(df_incendios, df_estaciones):
    """
    Asigna la estación meteorológica más cercana a cada incendio
    """
    print(" Asignando estación más cercana a cada incendio...")

    
    est_rad = np.deg2rad(df_estaciones[['lat_est', 'lon_est']].values)
    inc_rad = np.deg2rad(df_incendios[['lat', 'lon']].values)

    tree = BallTree(est_rad, metric='haversine')
    distancias, indices = tree.query(inc_rad, k=1)

   
    distancias_km = distancias.flatten() * 6371

    df_res = df_incendios.copy()
    idx_flat = indices.flatten()
    df_res['id_estacion'] = df_estaciones.iloc[idx_flat]['indicativo'].values
    df_res['nombre_estacion'] = df_estaciones.iloc[idx_flat]['nombre'].values
    df_res['distancia_estacion_km'] = distancias_km.round(2)

    print(f"    Estaciones asignadas")
    print(f"    Distancia media:  {distancias_km.mean():.1f} km")
    print(f"    Distancia máxima: {distancias_km.max():.1f} km")
    print(f"    P95 distancia:    {np.percentile(distancias_km, 95):.1f} km")

    return df_res




def _fmt_fecha(fecha):
    """Convierte fecha al formato AEMET: YYYY-MM-DDT00:00:00UTC"""
    if isinstance(fecha, str):
        fecha = pd.to_datetime(fecha)
    return fecha.strftime('%Y-%m-%dT00:00:00UTC')


def _descargar_clima_estacion(id_estacion, fecha_ini, fecha_fin, api_key):
    
    todos = []
    f_actual = pd.to_datetime(fecha_ini)
    f_final = pd.to_datetime(fecha_fin)

    while f_actual < f_final:
        f_chunk_fin = min(
            f_actual + pd.Timedelta(days=CHUNK_DAYS) - pd.Timedelta(days=1),
            f_final
        )
        endpoint = (
            f"valores/climatologicos/diarios/datos/"
            f"fechaini/{_fmt_fecha(f_actual)}/"
            f"fechafin/{_fmt_fecha(f_chunk_fin)}/"
            f"estacion/{id_estacion}"
        )
        datos = consultar_aemet(endpoint, api_key)
        if datos and isinstance(datos, list):
            todos.extend(datos)

        time.sleep(API_PAUSE)
        f_actual = f_chunk_fin + pd.Timedelta(days=1)

    return todos


def descargar_clima_todas_estaciones(estaciones_ids, fecha_ini, fecha_fin,
                                     api_key, ruta_cache=None):
    
    
    if ruta_cache is None:
        ruta_cache = os.path.join(
            os.path.dirname(__file__), '..', 'datos', 'external', 'cache_aemet'
        )
    os.makedirs(ruta_cache, exist_ok=True)

    todos = []
    unicas = sorted(set(estaciones_ids))

    print(f"\n  Descargando climatología diaria de {len(unicas)} estaciones...")
    print(f"   Periodo: {fecha_ini} → {fecha_fin}")
    print(f"   Cache:   {os.path.abspath(ruta_cache)}")

    for i, id_est in enumerate(tqdm(unicas, desc="Estaciones")):
        archivo_cache = os.path.join(ruta_cache, f"{id_est}.json")

        
        if os.path.exists(archivo_cache):
            with open(archivo_cache, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            todos.extend(datos)
            continue

        
        datos = _descargar_clima_estacion(id_est, fecha_ini, fecha_fin, api_key)

        if datos:
            with open(archivo_cache, 'w', encoding='utf-8') as f:
                json.dump(datos, f, ensure_ascii=False)
            todos.extend(datos)
        else:
            with open(archivo_cache, 'w', encoding='utf-8') as f:
                json.dump([], f)
            print(f"    Sin datos para estación {id_est}")

        
        if (i + 1) % 20 == 0:
            time.sleep(3)

    print(f"    {len(todos)} registros diarios descargados")
    return pd.DataFrame(todos) if todos else pd.DataFrame()




def _limpiar_valor(valor):
    """Limpia valores numéricos de AEMET (comas, 'Ip', 'Acum', etc.)."""
    if pd.isna(valor) or valor == '' or valor == 'Varias':
        return np.nan
    s = str(valor).strip()
    if s == 'Ip':      
        return 0.0
    if s == 'Acum':      
        return np.nan
    try:
        return float(s.replace(',', '.'))
    except ValueError:
        return np.nan


def limpiar_datos_clima(df_clima):
    
    if df_clima.empty:
        print(" DataFrame de clima vacío, nada que limpiar")
        return df_clima

    print("\n Limpiando datos climáticos...")

    s
    cols_disp = {k: v for k, v in COLUMNAS_CLIMA.items() if k in df_clima.columns}
    df = df_clima[list(cols_disp.keys())].rename(columns=cols_disp).copy()

    
    df['fecha_clima'] = pd.to_datetime(df['fecha_clima'], errors='coerce')

    
    cols_num = [
        'temp_media', 'temp_max', 'temp_min', 'precipitacion',
        'humedad_media', 'viento_medio', 'racha_max'
    ]
    for col in cols_num:
        if col in df.columns:
            df[col] = df[col].apply(_limpiar_valor)

    
    df = df.drop_duplicates(subset=['id_estacion', 'fecha_clima'])

    print(f"    {len(df)} registros climáticos limpios")
    print(f"    Completitud por variable:")
    for col in cols_num:
        if col in df.columns:
            pct = (1 - df[col].isna().mean()) * 100
            print(f"      {col}: {pct:.1f}%")

    return df




def cruzar_incendios_clima(df_incendios, df_clima):
    print("\n Cruzando datos de incendios con clima...")

    df_inc = df_incendios.copy()
    df_inc['fecha_ini'] = pd.to_datetime(df_inc['fecha_ini'])
    df_inc['fecha_cruce'] = df_inc['fecha_ini'].dt.normalize()

    df_cl = df_clima.copy()
    df_cl['fecha_cruce'] = pd.to_datetime(df_cl['fecha_clima']).dt.normalize()

    df_merged = df_inc.merge(
        df_cl,
        left_on=['id_estacion', 'fecha_cruce'],
        right_on=['id_estacion', 'fecha_cruce'],
        how='left'
    )
    df_merged = df_merged.drop(columns=['fecha_cruce', 'fecha_clima'], errors='ignore')

    total = len(df_merged)
    con_clima = df_merged['temp_media'].notna().sum()
    print(f"    Cruce completado")
    print(f"    Incendios con datos climáticos: {con_clima}/{total} "
          f"({con_clima/total*100:.1f}%)")

    return df_merged




def añadir_variables_previas(df_incendios, df_clima, dias_previos=7):
    """
    

    Añade columnas:
      - prec_acum_7d:   precipitación acumulada en los 7 días anteriores
      - tmax_max_7d:    temperatura máxima en los 7 días anteriores
      - dias_sin_lluvia: días consecutivos sin lluvia antes del incendio
    """
    print(f"\n Calculando variables de los {dias_previos} días previos...")

    
    df_cl = df_clima.copy()
    df_cl['fecha_clima'] = pd.to_datetime(df_cl['fecha_clima'])
    df_cl = df_cl.sort_values(['id_estacion', 'fecha_clima'])

    rolling_frames = []
    for estacion, grupo in df_cl.groupby('id_estacion'):
        g = grupo.set_index('fecha_clima').sort_index()

        
        idx = pd.date_range(g.index.min(), g.index.max(), freq='D')
        g = g.reindex(idx)
        g['id_estacion'] = estacion

        
        if 'precipitacion' in g.columns:
            g[f'prec_acum_{dias_previos}d'] = (
                g['precipitacion'].rolling(window=dias_previos, min_periods=1)
                .sum().shift(1)
            )
            
            llueve = (g['precipitacion'].fillna(0) > 0.1).astype(int)
            bloques = llueve.cumsum()
            g['dias_sin_lluvia'] = g.groupby(bloques).cumcount()
            
            g['dias_sin_lluvia'] = g['dias_sin_lluvia'].shift(1)

        
        if 'temp_max' in g.columns:
            g[f'tmax_max_{dias_previos}d'] = (
                g['temp_max'].rolling(window=dias_previos, min_periods=1)
                .max().shift(1)
            )

        g['fecha_cruce'] = g.index
        rolling_frames.append(g.reset_index(drop=True))

    df_rolling = pd.concat(rolling_frames, ignore_index=True)

    cols_nuevas = [
        'id_estacion', 'fecha_cruce',
        f'prec_acum_{dias_previos}d', 'dias_sin_lluvia',
        f'tmax_max_{dias_previos}d'
    ]
    cols_disponibles = [c for c in cols_nuevas if c in df_rolling.columns]
    df_rolling_sel = df_rolling[cols_disponibles].copy()

    df_inc = df_incendios.copy()
    df_inc['fecha_ini'] = pd.to_datetime(df_inc['fecha_ini'])
    df_inc['fecha_cruce'] = df_inc['fecha_ini'].dt.normalize()

    df_res = df_inc.merge(
        df_rolling_sel,
        on=['id_estacion', 'fecha_cruce'],
        how='left'
    )
    df_res = df_res.drop(columns=['fecha_cruce'], errors='ignore')

    print(f"    Variables previas calculadas")
    return df_res




def ejecutar_pipeline(ruta_incendios=None, ruta_salida=None, ruta_api=None):

    base = os.path.join(os.path.dirname(__file__), '..')

    if ruta_incendios is None:
        ruta_incendios = os.path.join(
            base, 'datos', 'processed', '01_incendios_CV_limpio.csv'
        )
    if ruta_salida is None:
        ruta_salida = os.path.join(
            base, 'datos', 'processed', '02_incendios_clima_CV.csv'
        )

    
    api_key = cargar_api_key(ruta_api)
    print(f" API key cargada: {api_key[:20]}...")

    # 2. Incendios
    print(f"\n Cargando incendios: {ruta_incendios}")
    df_inc = pd.read_csv(ruta_incendios, sep=';')
    df_inc['fecha_ini'] = pd.to_datetime(df_inc['fecha_ini'], dayfirst=True, format='mixed')
    df_inc['lat'] = pd.to_numeric(df_inc['lat'], errors='coerce')
    df_inc['lon'] = pd.to_numeric(df_inc['lon'], errors='coerce')
    df_inc = df_inc.dropna(subset=['lat', 'lon', 'fecha_ini'])
    print(f"   {len(df_inc)} incendios cargados")

    
    df_est = obtener_estaciones_cv(api_key)

    
    df_inc = asignar_estacion_cercana(df_inc, df_est)

    
    f_min = (df_inc['fecha_ini'].min() - pd.Timedelta(days=10)).strftime('%Y-01-01')
    f_max = df_inc['fecha_ini'].max().strftime('%Y-12-31')

    
    ids_unicos = df_inc['id_estacion'].unique().tolist()
    df_clima_raw = descargar_clima_todas_estaciones(
        ids_unicos, f_min, f_max, api_key
    )

    
    df_clima = limpiar_datos_clima(df_clima_raw)

    
    df_final = cruzar_incendios_clima(df_inc, df_clima)

    
    df_final = añadir_variables_previas(df_final, df_clima, dias_previos=7)

    
    print(f"\n Guardando resultado: {ruta_salida}")
    df_final.to_csv(ruta_salida, sep=';', index=False, encoding='utf-8-sig')
    print(f"    {len(df_final)} registros × {len(df_final.columns)} columnas")

    return df_final, df_est, df_clima


if __name__ == '__main__':
    ejecutar_pipeline()
