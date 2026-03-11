import openeo
from openeo.processes import process, if_
import geopandas as gpd
import json
import numpy as np
import time # Přidáno pro omezení rychlosti požadavků (rate limiting)

# ==========================================
# 1. Autentizace
# ==========================================
print("Připojování k backendu CDSE openEO...")
connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

# ==========================================
# 2. Příprava a reprojekce vektorových dat
# ==========================================
print("Načítání a příprava vektorových segmentů dálnic...")
vector_path = "highway_segments_100m_selected.geojson"
highway_gdf = gpd.read_file(vector_path)

# KRITICKÉ: Reprojekce do EPSG:4326 pro zajištění standardní kompatibility s openEO
if highway_gdf.crs != "EPSG:4326":
    highway_gdf = highway_gdf.to_crs("EPSG:4326")

# ==========================================
# 3. Definice logiky zpracování
# ==========================================
THRESHOLD = 0.02 

def detect_trucks_nd(x):
    """
    Logika normalizované diference pro detekci pohybujících se cílů.
    x obsahuje pouze [B02, B04] díky optimalizovanému načítání datové krychle.
    """
    b02 = x.array_element(index=0)
    b04 = x.array_element(index=1)
    
    # Matematické operátory jsou v Python klientovi openEO nativně přetíženy
    nd = (b02 - b04) / (b02 + b04)
    
    # Detekce přítomnosti kamionu
    is_truck = nd < THRESHOLD
    
    # Přetypování boolean na Double kvůli kompatibilitě se schématem Sparku během prostorové agregace
    return if_(is_truck, 1.0, 0.0)

def identify_clouds(x):
    scl = x.array_element(index=0)
    is_cloud = (scl == 1).or_(scl == 3).or_(scl == 8).or_(scl == 9).or_(scl == 10)
    return is_cloud

def identify_valid_pixels(x):
    """
    Inverze logiky oblačné masky: vrací 1.0 pro platné čisté pixely, 0.0 pro šum/oblačnost.
    """
    scl = x.array_element(index=0)
    is_cloud = (scl == 1).or_(scl == 3).or_(scl == 8).or_(scl == 9).or_(scl == 10)
    return if_(is_cloud, 0.0, 1.0)

temporal_extent = ["2025-01-01", "2025-12-31"]

# ==========================================
# 4. Strategie rozdělení na části (Chunking) 
# ==========================================
CHUNK_SIZE = 500
num_chunks = int(np.ceil(len(highway_gdf) / CHUNK_SIZE))
print(f"Dataset byl rozdělen na {num_chunks} částí (chunks) kvůli dodržení limitů API.")

# Přidání sloupců pro uložení výsledků
highway_gdf['truck_pixel_count'] = 0.0
highway_gdf['valid_pixel_count'] = 0.0

for i in range(num_chunks):
    print(f"\n--- Zpracovávám část {i + 1} z {num_chunks} ---")
    
    chunk_gdf = highway_gdf.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
    bounds = chunk_gdf.total_bounds
    spatial_extent = {
        "west": bounds[0], "south": bounds[1], 
        "east": bounds[2], "north": bounds[3]
    }
        
    # OPTIMALIZACE: B03 byl odstraněn z pole pásem
    datacube_l1c = connection.load_collection(
        "SENTINEL2_L1C",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["B02", "B04"]
    )
    
    datacube_l2a = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["SCL"]
    )
        
    # --- A. PIPELINE PRO DETEKCI KAMIONŮ ---
    cloud_mask = datacube_l2a.reduce_dimension(dimension="bands", reducer=identify_clouds)
    datacube_masked = datacube_l1c.mask(mask=cloud_mask)
    truck_mask = datacube_masked.reduce_dimension(dimension="bands", reducer=detect_trucks_nd)
    truck_mask_aligned = truck_mask.resample_spatial(resolution=10, projection=32633)
    truck_mask_2d = truck_mask_aligned.reduce_dimension(dimension="t", reducer="sum")
    
    # --- B. PIPELINE PRO VÝPOČET PLATNÝCH PIXELŮ ---
    valid_mask = datacube_l2a.reduce_dimension(dimension="bands", reducer=identify_valid_pixels)
    valid_mask_aligned = valid_mask.resample_spatial(resolution=10, projection=32633)
    valid_mask_2d = valid_mask_aligned.reduce_dimension(dimension="t", reducer="sum")

    # --- C. PROSTOROVÁ AGREGACE ---
    clean_chunk_gdf = chunk_gdf[['geometry']]
    chunk_geometries = json.loads(clean_chunk_gdf.to_json())
    
    zonal_stats_trucks = truck_mask_2d.aggregate_spatial(
        geometries=chunk_geometries,
        reducer="sum"
    )
    
    zonal_stats_valid = valid_mask_2d.aggregate_spatial(
        geometries=chunk_geometries,
        reducer="sum"
    )
    
    print("Spouštím prostorové agregace...")
    try:
        # Provedení obou prostorových agregací (seznamy mapující 1:1 na vstupní geometrie)
        results_trucks = zonal_stats_trucks.execute() 
        results_valid = zonal_stats_valid.execute()
        
        for feature_index in range(len(chunk_gdf)):
            # Defenzivní rozbalení vnořených seznamů pro počet kamionů
            val_t = results_trucks[feature_index]
            while isinstance(val_t, list):
                val_t = val_t[0] if len(val_t) > 0 else 0.0
            if val_t is None: val_t = 0.0
            
            # Defenzivní rozbalení vnořených seznamů pro počet platných pixelů
            val_v = results_valid[feature_index]
            while isinstance(val_v, list):
                val_v = val_v[0] if len(val_v) > 0 else 0.0
            if val_v is None: val_v = 0.0
                
            # Zpětné mapování na absolutní index hlavního highway_gdf
            absolute_index = chunk_gdf.index[feature_index]
            highway_gdf.at[absolute_index, 'truck_pixel_count'] = val_t
            highway_gdf.at[absolute_index, 'valid_pixel_count'] = val_v
            
    except Exception as e:
        print(f"Chyba při zpracování části {i+1}: {e}")
        
    # Zpoždění pro omezení rychlosti požadavků (rate limiting) jako ochrana proti chybám WAF 429
    time.sleep(10)

# ==========================================
# 5. Export výsledků
# ==========================================
output_file = "highway_segments_processed.geojson"
highway_gdf.to_file(output_file, driver="GeoJSON")
print(f"\nZpracování dokončeno. Výsledky uloženy do {output_file}")
