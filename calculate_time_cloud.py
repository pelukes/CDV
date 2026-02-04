# -*- coding: utf-8 -*-
import os
import pystac_client
import planetary_computer
import pandas as pd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from datetime import datetime, timedelta, timezone

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
INPUT_CSV = "S2_sensing_dates.csv"   
OUTPUT_CSV = "S2_sensing_dates_updated.csv"
OUTPUT_DIR = "output_time_rasters_float"

# Paths to your local master masks
MASK_FILES = {
    "S2A": "masks/S2A_MSK_DETFOO_B02.jp2",
    "S2B": "masks/S2B_MSK_DETFOO_B02.jp2",
    "S2C": "masks/S2C_MSK_DETFOO_B02.jp2"
}

# Physics Constants (ISPRS 2022)
TIME_PER_LINE_SECONDS = 0.001515 
DETECTOR_PARALLAX_SECONDS = 2.6 

# CALIBRATION: Define which detectors are "Trailing" (delayed by 2.6s)
# For S2A/B, typically the ODD detectors are trailing in the focal plane.
# For S2C, due to 180 deg rotation, the parity is usually flipped.
PARALLAX_CONFIG = {
    "S2A": "ODD", 
    "S2B": "ODD", 
    "S2C": "EVEN" 
}

# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------

def parse_scene_string(scene_str):
    parts = scene_str.split('_')
    date_part = parts[0][:8] 
    mgrs_tile = parts[-1][1:] if parts[-1].startswith('T') else parts[-1]
    date_formatted = f"{date_part[0:4]}-{date_part[4:6]}-{date_part[6:8]}"
    return date_formatted, mgrs_tile

def get_stac_metadata(q_date, q_tile):
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        datetime=q_date,
        query={"s2:mgrs_tile": {"eq": q_tile}}
    )
    items = list(search.items())
    return items[0] if items else None

# -------------------------------------------------------------------------
# CORE LOGIC
# -------------------------------------------------------------------------

def process_hybrid(scene_id, output_folder):
    print(f"\nProcessing: {scene_id}")
    
    # 1. Fetch Metadata from PC
    q_date, q_tile = parse_scene_string(scene_id)
    item = get_stac_metadata(q_date, q_tile)
    if not item:
        print("  ! Error: Metadata not found.")
        return None
    
    # 2. Identify Sensor
    raw_plat = item.properties.get("platform", "")
    if "2A" in raw_plat.upper(): sensor_id = "S2A"
    elif "2B" in raw_plat.upper(): sensor_id = "S2B"
    elif "2C" in raw_plat.upper(): sensor_id = "S2C"
    else: sensor_id = item.id[:3].upper() 

    # 3. Load Mask and Parallax Calibration
    mask_path = MASK_FILES.get(sensor_id)
    parity_mode = PARALLAX_CONFIG.get(sensor_id, "ODD")
    
    print(f"  -> Sensor: {sensor_id} | Trailing Detectors: {parity_mode}")

    if not mask_path or not os.path.exists(mask_path):
        print(f"  ! Error: Mask missing for {sensor_id}")
        return None

    # Time Setup
    base_time = item.datetime.replace(tzinfo=timezone.utc)
    base_time_unix = base_time.timestamp()

    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)
        
        # Apply Calibration Logic
        if parity_mode == "ODD":
            trailing_mask = (mask_data % 2 != 0).astype(np.float32)
        else:
            # Shift the 2.6s to EVEN detectors for S2C
            trailing_mask = (mask_data % 2 == 0).astype(np.float32)
        
        h, w = mask_data.shape
        final_crs = src.crs
        final_transform = src.transform

        # Upscale check
        if w < 5000:
            print(f"  ~ Upscaling {sensor_id} mask...")
            scale = 10980 / w
            final_transform = Affine(src.transform.a/scale, 0, src.transform.c, 
                                     0, src.transform.e/scale, src.transform.f)
            with rasterio.open('mem','w',driver='MEM',height=h,width=w,count=1,dtype='float32',transform=src.transform,crs=src.crs) as mem:
                mem.write(trailing_mask, 1)
                trailing_mask = mem.read(1, out_shape=(10980,10980), resampling=Resampling.bilinear)
            h, w = 10980, 10980

    # 4. Calculation
    row_indices = np.indices((h, w))[0]
    time_offset = trailing_mask * DETECTOR_PARALLAX_SECONDS
    ts_unix = base_time_unix + (row_indices * TIME_PER_LINE_SECONDS) + time_offset

    # HHMMSS.ssss
    secs_midnight = ts_unix % 86400
    hhmmss = (np.floor(secs_midnight/3600)*10000) + (np.floor((secs_midnight%3600)/60)*100) + (secs_midnight%60)

    # 5. Save
    out_path = os.path.join(output_folder, f"{scene_id}_HHMMSS.tif")
    with rasterio.open(out_path, 'w', driver='GTiff', height=h, width=w, count=1, dtype='float64', crs=final_crs, transform=final_transform, compress='lzw') as dst:
        dst.write(hhmmss, 1)
        dst.update_tags(comment=f"Sensor={sensor_id}, Mode={parity_mode}_Trailing")
    
    print("  Success.")
    return base_time.isoformat()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    df['Scene acquisition time'] = [process_hybrid(row['Scenes'], OUTPUT_DIR) for _, row in df.iterrows()]
    df.to_csv(OUTPUT_CSV, index=False)
