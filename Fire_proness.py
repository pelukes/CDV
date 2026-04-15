import os
import requests
from datetime import date, datetime, timedelta
import numpy as np
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import openeo
import cdsapi

# --------------------------------------------------
# 1. User Settings & Thresholds
# --------------------------------------------------
TARGET_YEAR = 2025
HISTORICAL_YEARS = range(2022, 2024) # 2018-2024 forms the baseline for 2025
MAX_CLOUD_COVER = 20
WINDOW_HALF_SIZE_DAYS = 15

# AOI: Central Czechia (Vysočina)
# STAC/OpenEO expects [West, South, East, North] or dict
bbox = [15.00, 49.00, 15.70, 49.45]
spatial_extent = {
    "west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3], "crs": "EPSG:4326"
}

# Local afternoon hours in UTC (CEST = UTC+2)
afternoon_hours_utc = ["11:00", "12:00", "13:00", "14:00"]

# Thresholds
ndwi_fire_prone_thr = -1.0
ndwi_very_fire_prone_thr = -1.5
vpd_fire_prone_thr_kpa = 1.5
vpd_very_fire_prone_thr_kpa = 2.0

# Output parameters
era5_annual_nc = f"era5land_afternoon_{TARGET_YEAR}.nc"
out_timeseries_tif = f"fire_proneness_timeseries_{TARGET_YEAR}.tif"

# --------------------------------------------------
# 2. Helper Functions
# --------------------------------------------------
def get_cloud_free_s2_dates(bbox, year, max_cloud):
    """Queries the new CDSE STAC v1 API for Sentinel-2 L2A dates."""
    print(f"Querying STAC v1 API for valid dates in {year}...")
    
    # NEW ENDPOINT
    stac_url = "https://stac.dataspace.copernicus.eu/v1/search"
    
    payload = {
        # NEW NATIVE COLLECTION ID
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
        # The new endpoint natively supports cloud cover queries without crashing
        "query": {
            "eo:cloud_cover": {"lte": max_cloud}
        },
        "limit": 100 
    }
    
    response = requests.post(stac_url, json=payload)
    
    if response.status_code != 200:
        print(f"STAC API Error Details: {response.text}")
    response.raise_for_status()
    
    items = response.json().get("features", [])
    
    valid_dates = set()
    for item in items:
        dt_str = item["properties"]["datetime"]
        # Extract just the date string (YYYY-MM-DD) to avoid microsecond parsing issues
        date_obj = datetime.strptime(dt_str[:10], "%Y-%m-%d").date()
        valid_dates.add(date_obj)
    
    sorted_dates = sorted(list(valid_dates))
    print(f"Found {len(sorted_dates)} valid L2A scenes with <= {max_cloud}% cloud cover.")
    return sorted_dates

def download_era5_annual_afternoon(year: int, area, out_nc: str):
    """Downloads a full year of afternoon ERA5-Land data in a single optimized request."""
    if os.path.exists(out_nc):
        print(f"Using existing ERA5-Land annual file: {out_nc}")
        return out_nc

    print(f"Downloading annual ERA5-Land data for {year}...")
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-land",
        {
            "variable": ["2m_temperature", "2m_dewpoint_temperature"],
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": afternoon_hours_utc,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": area, # N, W, S, E
        },
        out_nc,
    )
    return out_nc

def get_masked_ndwi(cube):
    """Computes NDWI and applies SCL mask via parallel tracks to avoid Graph Overlap errors."""
    scl = cube.band("SCL")
    mask = ~((scl == 4) | (scl == 5))
    
    b8a = cube.band("B8A")
    b11 = cube.band("B11")
    ndwi = (b8a - b11) / (b8a + b11)
    
    return ndwi.mask(mask)

def build_yearly_intervals(center_date, half_window, years):
    intervals = []
    for y in years:
        center = date(y, center_date.month, center_date.day)
        start = center - timedelta(days=half_window)
        end = center + timedelta(days=half_window + 1)
        intervals.append([start.isoformat(), end.isoformat()])
    return intervals

# --------------------------------------------------
# 3. Main Workflow Execution
# --------------------------------------------------
if __name__ == "__main__":
    
    # 3.1 Fetch available dates and ERA5 data
    valid_dates = get_cloud_free_s2_dates(bbox, TARGET_YEAR, MAX_CLOUD_COVER)
    
    if not valid_dates:
        print("No valid scenes found. Exiting.")
        exit()
        
    cds_area = [bbox[3], bbox[0], bbox[1], bbox[2]] # North, West, South, East
    download_era5_annual_afternoon(TARGET_YEAR, cds_area, era5_annual_nc)
    
    # Pre-load ERA5-Land locally
    ds_era5 = xr.open_dataset(era5_annual_nc)
    if "longitude" in ds_era5.dims:
        ds_era5 = ds_era5.rename({"longitude": "x", "latitude": "y"})
        
    # Calculate global VPD for the year
    t_c = ds_era5["t2m"] - 273.15
    td_c = ds_era5["d2m"] - 273.15
    es = 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))
    ea = 0.6108 * np.exp((17.27 * td_c) / (td_c + 237.3))
    vpd_annual = es - ea
    vpd_annual = vpd_annual.rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim="x", y_dim="y")
    time_dim = "valid_time" if "valid_time" in vpd_annual.dims else "time"
    
    # 3.2 Initialize openEO
    conn = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
    
    classified_scenes = []
    processed_dates = []

    # 3.3 Iterate through valid scenes
    for target_date in valid_dates:
        print(f"\n--- Processing Date: {target_date.isoformat()} ---")
        
        # --- A. OpenEO NDWI Anomaly Calculation ---
        hist_intervals = build_yearly_intervals(target_date, WINDOW_HALF_SIZE_DAYS, HISTORICAL_YEARS)
        
        # Load Baseline
        cube_hist = conn.load_collection(
            "SENTINEL2_L2A",
            spatial_extent=spatial_extent,
            temporal_extent=[hist_intervals[0][0], hist_intervals[-1][1]],
            bands=["B8A", "B11", "SCL"]
        )
        ndwi_hist = get_masked_ndwi(cube_hist)
        yearly_ndwi = ndwi_hist.aggregate_temporal(intervals=hist_intervals, reducer="median")
        
        baseline_mean = yearly_ndwi.reduce_dimension(dimension="t", reducer="mean")
        baseline_sd = yearly_ndwi.reduce_dimension(dimension="t", reducer="sd") + 1e-6
        
        # Load Current
        cur_start = target_date - timedelta(days=WINDOW_HALF_SIZE_DAYS)
        cur_end = target_date + timedelta(days=WINDOW_HALF_SIZE_DAYS + 1)
        cube_current = conn.load_collection(
            "SENTINEL2_L2A",
            spatial_extent=spatial_extent,
            temporal_extent=[cur_start.isoformat(), cur_end.isoformat()],
            bands=["B8A", "B11", "SCL"]
        )
        
        current_ndwi = get_masked_ndwi(cube_current)
        current_median = current_ndwi.reduce_dimension(dimension="t", reducer="median")
        
        # Compute and Export Anomaly
        z_score = (current_median - baseline_mean) / baseline_sd
        z_score_export = z_score.add_dimension(name="bands", label="z_score", type="bands")
        
        tmp_tif = f"temp_ndwi_z_{target_date.isoformat()}.tif"
        
        print(f"Submitting asynchronous job to CDSE backend for {target_date}...")
        # execute_batch automatically creates, starts, waits for, and downloads the job
        z_score_export.execute_batch(
            outputfile=tmp_tif, 
            out_format="GTiff",
            title=f"NDWI_Anomaly_{target_date.isoformat()}"
        )
        
        # --- B. Local VPD Extraction & Classification ---
        # Slice ERA5 data strictly for this day and find the afternoon maximum
        date_str = target_date.isoformat()
        vpd_daily = vpd_annual.sel({time_dim: slice(f"{date_str} 00:00:00", f"{date_str} 23:59:59")})
        vpd_max = vpd_daily.max(dim=time_dim)
        
        # THE FIX: Use a context manager to open, load into RAM, and automatically close the file handle
        with rioxarray.open_rasterio(tmp_tif) as src:
            ndwi_da = src.squeeze(drop=True).load()
            
        # Reproject VPD
        vpd_matched = vpd_max.rio.reproject_match(ndwi_da, resampling=Resampling.bilinear)
        
        # Matrix boolean operations
        valid = np.isfinite(ndwi_da) & np.isfinite(vpd_matched)
        fire_prone = (ndwi_da <= ndwi_fire_prone_thr) & (vpd_matched >= vpd_fire_prone_thr_kpa)
        very_fire_prone = (ndwi_da <= ndwi_very_fire_prone_thr) & (vpd_matched >= vpd_very_fire_prone_thr_kpa)
        
        # Reclassify to Integers
        out_class = xr.full_like(ndwi_da, 0, dtype=np.uint8)
        out_class = xr.where(valid, 1, out_class)
        out_class = xr.where(valid & fire_prone, 2, out_class)
        out_class = xr.where(valid & very_fire_prone, 3, out_class)
        
        # Assign time dimension strictly for stacking 
        # THE FIX: Add 'ns' (nanoseconds) to silence the xarray UserWarning
        out_class = out_class.expand_dims(time=[np.datetime64(target_date, 'ns')])
        out_class = out_class.rio.write_crs(ndwi_da.rio.crs)
        
        classified_scenes.append(out_class)
        processed_dates.append(date_str)
        
        # Cleanup temporary TIFF to save disk space
        # (This will now succeed because the 'with' block closed the file)
        os.remove(tmp_tif)

    # 3.4 Compile Multi-Temporal GeoTIFF
    print("\nCompiling final multi-band GeoTIFF time series...")
    time_series_da = xr.concat(classified_scenes, dim="time")
    
    # Ensure attributes are clean before rasterization
    time_series_da.name = "fire_proneness"
    time_series_da.attrs["long_name"] = "0=NoData, 1=Safe, 2=FireProne, 3=Extreme"
    
    # rioxarray writes dimensions in (band, y, x) natively, mapping 'time' to 'band'
    time_series_da.rio.to_raster(
        out_timeseries_tif, 
        tags={"DATES": ",".join(processed_dates)}
    )
    
    print(f"Success. Annual time series saved to: {out_timeseries_tif}")
