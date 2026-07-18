import pandas as pd
import geopandas as gpd
import os
import glob
from pathlib import Path

def main():
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATOS_PROCESSED = BASE_DIR / "datos" / "processed"
    DATOS_EXTERNAL = BASE_DIR / "datos" / "external" / "vegetacion_cv"
    
    
    csv_path = DATOS_PROCESSED / "05_dataset_final_para_EDA.csv"
    salida_path = DATOS_PROCESSED / "06_dataset_final_enriquecido.csv"

    print(f"Cargando dataset base desde: {csv_path}")
    df = pd.read_csv(csv_path, sep=';', on_bad_lines='warn')

    
    gdf_puntos = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df['lon'], df['lat']),
        crs="EPSG:4326"
    )

    
    if not DATOS_EXTERNAL.exists():
        DATOS_EXTERNAL.mkdir(parents=True, exist_ok=True)
        print(f"ATENCIÓN: Creada carpeta {DATOS_EXTERNAL}. Por favor, pon el mapa ahí y vuelve a ejecutar.")
        return

    archivos_vectoriales = list(DATOS_EXTERNAL.rglob("*.gpkg"))
    if not archivos_vectoriales:
        archivos_vectoriales = list(DATOS_EXTERNAL.rglob("*.shp"))

    if not archivos_vectoriales:
        print(f"ERROR: No se encontró ningún archivo .gpkg ni .shp en {DATOS_EXTERNAL}")
        return

    mapa_path = archivos_vectoriales[0]
    print(f"Cargando mapa de vegetación: {mapa_path.name}")
    
    
    gdf_veg = gpd.read_file(mapa_path, engine='pyogrio')

    
    if gdf_veg.crs is None or gdf_veg.crs.to_string() != "EPSG:4326":
        print(f"Reproyectando mapa forestal de {gdf_veg.crs} a EPSG:4326...")
        gdf_veg = gdf_veg.to_crs("EPSG:4326")

    
    COLUMNA_VEGETACION_ORIGINAL = 'clase'
    
    if COLUMNA_VEGETACION_ORIGINAL not in gdf_veg.columns:
        print(f"ADVERTENCIA: No se encontró la columna '{COLUMNA_VEGETACION_ORIGINAL}' en el archivo.")
        print(f"Columnas disponibles: {list(gdf_veg.columns)}")
        for c in gdf_veg.columns:
            if gdf_veg[c].dtype == 'O' and c != 'geometry':
                COLUMNA_VEGETACION_ORIGINAL = c
                break

    print(f"Usando la columna '{COLUMNA_VEGETACION_ORIGINAL}' para clasificar.")

    
    print("Realizando cruce espacial (Spatial Join)...")
    cruce = gpd.sjoin(gdf_puntos, gdf_veg, how='left', predicate='intersects')

    
    def categorizar_vegetacion(valor):
        if pd.isna(valor):
            return "Urbano/Antropizado" 
            
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
        elif 'urbano' in v or 'improductivo' in v or 'construido' in v or 'vial' in v or 'playa' in v or 'roquedo' in v or 'escollera' in v or 'marisma' in v or 'río' in v or 'húmeda' in v or 'humeda' in v:
            return "Urbano/Otros"
        else:
            return "Urbano/Otros"

    
    if COLUMNA_VEGETACION_ORIGINAL in cruce.columns:
        cruce['tipo_vegetacion'] = cruce[COLUMNA_VEGETACION_ORIGINAL].apply(categorizar_vegetacion)
    else:
        
        cruce['tipo_vegetacion'] = "Urbano/Antropizado"

    
    cruce['tipo_vegetacion'] = cruce['tipo_vegetacion'].fillna('Urbano/Antropizado')

    
    cols_a_borrar = [c for c in gdf_veg.columns if c != 'geometry'] + ['index_right']
    cruce = cruce.drop(columns=[c for c in cols_a_borrar if c in cruce.columns], errors='ignore')
    
    
    df_final = pd.DataFrame(cruce.drop(columns=['geometry']))

    
    print(f"Guardando dataset final enriquecido en: {salida_path}")
    try:
        df_final.to_csv(salida_path, sep=';', index=False)
        print("Proceso completado con éxito. Fase 1 cerrada.")
    except PermissionError:
        print(f"ADVERTENCIA: No se pudo sobrescribir {salida_path.name} (¿abierto en Excel?).")
        fallback_path = salida_path.parent / (salida_path.stem + "_v2" + salida_path.suffix)
        print(f"Guardando como {fallback_path.name} en su lugar...")
        df_final.to_csv(fallback_path, sep=';', index=False)
        print("¡Proceso completado con éxito! Fase 1 cerrada.")

if __name__ == "__main__":
    main()
