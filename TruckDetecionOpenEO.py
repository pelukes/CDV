import openeo
import xarray as xr
import rioxarray
import numpy as np
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box
from rasterio import features
from s2cloudless import S2PixelCloudDetector

# ==========================================
# 1. OpenEO Split-Resolution Data Retrieval
# ==========================================
print("Querying OpenEO backend...")
conn = openeo.connect("https://openeofed.dataspace.copernicus.eu").authenticate_oidc()

roi = {"west": 16.5, "south": 49.1, "east": 16.7, "north": 49.3}
time_window = ["2025-04-01", "2025-08-31"]

# Define EPSG:32633 (UTM Zone 33N) to ensure metric units for the 30m buffer
TARGET_CRS = 32633 

# A. Fetch 60m Data (10 bands for s2cloudless)
print("Downloading 60m dataset for cloud masking...")
bands_cloud = ["B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"]
cube_60m = conn.load_collection(
    "SENTINEL2_L1C",
    spatial_extent=roi,
    temporal_extent=time_window,
    bands=bands_cloud
).resample_spatial(resolution=60, projection=TARGET_CRS) # Enforce UTM
cube_60m.download("s2_60m_clouds.nc", format="NetCDF")

# B. Fetch 10m Data (B02 and B04 for Parallax)
print("Downloading 10m dataset for parallax analysis...")
bands_parallax = ["B02", "B04", "SCL"]
cube_10m = conn.load_collection(
    "SENTINEL2_L2A",
    spatial_extent=roi,
    temporal_extent=time_window,
    bands=bands_parallax
).resample_spatial(resolution=10, projection=TARGET_CRS) # Enforce UTM
cube_10m.download("s2_10m_parallax.nc", format="NetCDF")

# ==========================================
# 2. Local Cloud Masking (at 60m)
# ==========================================
print("Running local s2cloudless on 60m data...")
# Use rioxarray to read the CRS automatically
ds_60 = rioxarray.open_rasterio("s2_60m_clouds.nc", decode_coords="all")

stacked_data = np.stack([ds_60[b].values for b in bands_cloud], axis=-1)
reflectances = np.clip(stacked_data.astype(np.float32) / 10000.0, 0.0, 1.0)

detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2, all_bands=False)
cloud_mask_60m = detector.get_cloud_masks(reflectances)
cloud_probs_60m = detector.get_cloud_probability_maps(reflectances)

# Preserve xarray structure
mask_60m_da = xr.DataArray(cloud_mask_60m, coords=[ds_60.t, ds_60.y, ds_60.x], dims=["t", "y", "x"])
probs_60m_da = xr.DataArray(cloud_probs_60m, coords=[ds_60.t, ds_60.y, ds_60.x], dims=["t", "y", "x"])

# ==========================================
# 3. Vector Data: Fetching and Buffering Roads
# ==========================================
print("Fetching OpenStreetMap road networks and applying 10m buffer...")

# 1. Increase the Overpass API timeout to 10 minutes (600 seconds)
ox.settings.timeout = 600 
ox.settings.log_console = True
ox.settings.use_cache = True 

osm_bbox = box(roi["west"], roi["south"], roi["east"], roi["north"])

# 2. Strictly limit to major thoroughfares. 
# REMOVED: "residential" and "unclassified" to prevent server timeouts.
major_roads = [
    "motorway", "motorway_link", 
    "trunk", "trunk_link", 
    "primary", "primary_link"
]

try:
    roads_gdf = ox.features_from_polygon(osm_bbox, tags={"highway": major_roads})
except AttributeError:
    roads_gdf = ox.geometries_from_polygon(osm_bbox, tags={"highway": major_roads})

# Filter for strictly linear geometry (ignore point nodes)
roads_gdf = roads_gdf[roads_gdf.geometry.type.isin(['LineString', 'MultiLineString'])]

# Reproject to UTM Zone 33N to match the raster data (metric units)
roads_gdf = roads_gdf.to_crs(epsg=TARGET_CRS)

# Apply the 30m buffer
print(f"Buffering {len(roads_gdf)} road segments by 10 meters...")
buffered_roads = roads_gdf.geometry.buffer(10)

# ==========================================
# 4. Rasterize Road Buffer and Align Grids
# ==========================================
print("Rasterizing road buffer to match the 10m grid...")
ds_10 = rioxarray.open_rasterio("s2_10m_parallax.nc", decode_coords="all")

# Upsample the 60m cloud mask to the 10m grid
cloud_mask_10m = mask_60m_da.interp_like(ds_10, method="nearest")
cloud_probs_10m = probs_60m_da.interp_like(ds_10, method="nearest")

# Rasterize the buffered vector polygons into a 2D array matching the 10m spatial grid
# 1 = inside buffer, 0 = outside buffer
road_mask_2d = features.rasterize(
    shapes=buffered_roads.values,
    out_shape=(len(ds_10.y), len(ds_10.x)),
    transform=ds_10.rio.transform(),
    fill=0,
    default_value=1,
    dtype=np.uint8
)

# ==========================================
# 5. Compound Masking and Parallax Detection
# ==========================================
print("Applying compound mask (s2cloudless + SCL + Roads) and computing anomalies...")

b02_10m = ds_10["B02"].values
b04_10m = ds_10["B04"].values
scl_10m = ds_10["SCL"].values 

# 1. Calculate the RAW Parallax Anomaly across the entire 10m grid
parallax_index_raw = (b02_10m - b04_10m) / (b02_10m + b04_10m + 1e-8)

# 2. Define invalid SCL classes
# 3 = Cloud Shadows, 8 = Cloud Medium Prob, 9 = Cloud High Prob, 10 = Thin Cirrus
invalid_scl_classes = [3, 8, 9, 10]
scl_invalid_mask = np.isin(scl_10m, invalid_scl_classes)

# 3. Create the FINAL COMBINED CLOUD MASK (Atmosphere only)
# 1 = Invalid (Cloud or Shadow), 0 = Valid (Clear)
# We use bitwise OR (|) because if EITHER s2cloudless OR SCL flags it, we want it masked.
combined_cloud_mask = (cloud_mask_10m.values == 1) | scl_invalid_mask

# 4. Create the valid pixel mask for the parallax math
# Must be CLEAR in the combined mask (0) AND Must be inside the road buffer (1)
valid_mask = (combined_cloud_mask == 0) & (road_mask_2d == 1)

# 5. Apply the mask to the parallax index
parallax_index_masked = np.where(valid_mask, parallax_index_raw, np.nan)

# 6. Apply the threshold to find moving targets
PARALLAX_THRESHOLD = 0.15
moving_targets = np.where(np.abs(parallax_index_masked) > PARALLAX_THRESHOLD, 1, 0)

# ==========================================
# 6. Save Final 10m Results
# ==========================================
# Append the generated arrays back to the dataset
ds_10["moving_targets"] = (("t", "y", "x"), moving_targets)

# Save the original individual masks if needed for debugging
ds_10["s2cloudless_mask"] = cloud_mask_10m 

# ADDED: Save the final, combined atmospheric mask (s2cloudless + SCL)
# Storing as uint8 to save space (1 = cloud/shadow, 0 = clear)
ds_10["combined_cloud_mask"] = (("t", "y", "x"), combined_cloud_mask.astype(np.uint8))

ds_10["road_buffer_mask"] = (("y", "x"), road_mask_2d) 
ds_10["parallax_index_raw"] = (("t", "y", "x"), parallax_index_raw)

# Export to a final analytical NetCDF
ds_10.to_netcdf("s2_analyzed_results_10m.nc")
print("Pipeline complete. Masked high-resolution results saved to s2_analyzed_results_10m.nc")
