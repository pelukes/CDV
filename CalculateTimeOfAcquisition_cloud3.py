# -*- coding: utf-8 -*-
import os
import requests
import xml.etree.ElementTree as ET
import pystac_client
import planetary_computer
import pandas as pd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from datetime import datetime, timedelta

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
# Define the spatio-temporal and environmental parameters for the STAC query
START_DATE = "2025-03-19"
END_DATE = "2025-03-21"
TARGET_TILE = "33UXQ"
MAX_CLOUD_COVER = 20.0  # Maximum allowed cloud cover percentage (0-100)

OUTPUT_CSV = f"S2_sensing_dates_{TARGET_TILE}_{START_DATE}_to_{END_DATE}_CC{int(MAX_CLOUD_COVER)}.csv"
OUTPUT_DIR = f"output_time_rasters_{TARGET_TILE}"

# Paths to your local master masks
MASK_FILES = {
    "S2A": "masks/S2A_MSK_DETFOO_B02.jp2",
    "S2B": "masks/S2B_MSK_DETFOO_B02.jp2",
    "S2C": "masks/S2C_MSK_DETFOO_B02.jp2" 
}

# Physics Constants (ISPRS 2022)
TIME_PER_LINE_SECONDS = 0.001515 
DETECTOR_PARALLAX_SECONDS = 2.6 

# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------

def fetch_stac_items(start_date, end_date, tile, max_cloud):
    """Queries PC for S2 L2A scenes matching date, tile, and cloud cover criteria."""
    print(f"Querying STAC catalog for tile {tile} between {start_date} and {end_date}...")
    print(f"Applying cloud cover filter: <= {max_cloud}%")
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    # The STAC query limits results server-side based on the 'eo:cloud_cover' property
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        datetime=f"{start_date}/{end_date}",
        query={
            "s2:mgrs_tile": {"eq": tile},
            "eo:cloud_cover": {"lte": max_cloud}
        }
    )
    
    items = list(search.items())
    print(f"Found {len(items)} scenes matching all criteria.")
    return items

def parse_cloud_xml(xml_url):
    response = requests.get(xml_url, timeout=15)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    
    # 1. Exact Sensing Time
    sensing_time_str = root.find('.//SENSING_TIME').text
    mean_sensing_time = datetime.strptime(sensing_time_str, '%Y-%m-%dT%H:%M:%S.%fZ')
    
    # 2. Geospatial Framework
    geoposition = root.find('.//Geoposition[@resolution="10"]')
    ulx = float(geoposition.findtext('ULX'))
    uly = float(geoposition.findtext('ULY'))
    xdim = float(geoposition.findtext('XDIM'))
    ydim = abs(float(geoposition.findtext('YDIM')))
    
    size = root.find('.//Size[@resolution="10"]')
    rows = int(size.findtext('NROWS'))
    cols = int(size.findtext('NCOLS'))
    
    crs_code = root.find('.//HORIZONTAL_CS_CODE').text
    
    return mean_sensing_time, ulx, uly, xdim, ydim, rows, cols, crs_code

# -------------------------------------------------------------------------
# CORE LOGIC
# -------------------------------------------------------------------------

def process_stac_item(item, output_folder):
    scene_id = item.id
    print(f"\nProcessing: {scene_id}")
    
    # Extract Metadata Attributes
    cloud_cover = item.properties.get("eo:cloud_cover", np.nan)
    relative_orbit = item.properties.get("sat:relative_orbit", np.nan)
    print(f"  -> Cloud Cover: {cloud_cover}%")
    print(f"  -> Relative Orbit: {relative_orbit}")
    
    # 1. Determine Satellite Platform rigorously from STAC metadata
    platform = item.properties.get("platform", "").upper()
    if "2A" in platform:
        satellite = "S2A"
    elif "2B" in platform:
        satellite = "S2B"
    elif "2C" in platform:
        satellite = "S2C"
    else:
        print(f"  ! Error: Could not determine satellite platform from metadata: {platform}")
        return None
        
    print(f"  -> Platform detected: {satellite}")

    # 2. Parse XML directly from cloud asset
    try:
        xml_url = item.assets["granule-metadata"].href
        print("  -> Fetching MTD_TL.xml from granule asset...")
        parsed_data = parse_cloud_xml(xml_url)
        mean_sensing_time, ulx, uly, xdim, ydim, rows, cols, crs_code = parsed_data
    except KeyError:
        print("  ! Error: 'granule-metadata' asset missing in STAC item.")
        return None
    except Exception as e:
        print(f"  ! Error retrieving or parsing XML: {e}")
        return None

    # 3. Prepare Time Variables
    total_line_duration = rows * TIME_PER_LINE_SECONDS
    start_time = mean_sensing_time - timedelta(seconds=total_line_duration / 2)
    start_sec = (start_time.hour * 3600) + (start_time.minute * 60) + start_time.second + (start_time.microsecond / 1e6)

    # 4. Load, Resample, and Binarize the Parallax Mask
    mask_path = MASK_FILES.get(satellite)
    if not mask_path or not os.path.exists(mask_path):
        print(f"  ! Error: Mask file for {satellite} not found at {mask_path}.")
        return None

    print(f"  -> Applying detector parallax mask for {satellite}...")
    with rasterio.open(mask_path) as src:
        # STRICT REQUIREMENT: Categorical data must use Nearest Neighbor resampling.
        raw_detector_mask = src.read(
            1, 
            out_shape=(rows, cols), 
            resampling=Resampling.nearest
        )

    # Resolve odd/even focal plane parity based on the platform
    print("  -> Resolving focal plane parity (Odd/Even)...")
    if satellite in ["S2A", "S2B"]:
        # For S2A/B, odd detectors (1, 3, 5...) trail the even detectors.
        trailing_mask = (raw_detector_mask % 2 != 0).astype(np.float64)
    elif satellite == "S2C":
        # For S2C, due to the 180-degree rotation, parity is flipped. 
        # Even detectors (2, 4, 6...) are trailing. Exclude 0 (nodata).
        trailing_mask = ((raw_detector_mask % 2 == 0) & (raw_detector_mask > 0)).astype(np.float64)
    else:
        trailing_mask = np.zeros((rows, cols), dtype=np.float64)

    # 5. Vectorized Matrix Calculation
    print("  -> Calculating pixel acquisition times...")
    row_indices = np.arange(rows, dtype=np.float64)[:, np.newaxis]
    time_offset = trailing_mask * DETECTOR_PARALLAX_SECONDS
    pixel_seconds_of_day = start_sec + (row_indices * TIME_PER_LINE_SECONDS) + time_offset

    hh = np.floor(pixel_seconds_of_day / 3600)
    mm = np.floor((pixel_seconds_of_day % 3600) / 60)
    ss = pixel_seconds_of_day % 60
    
    dn_values = (hh * 10000) + (mm * 100) + ss

    # 6. Save to GeoTIFF
    out_path = os.path.join(output_folder, f"{scene_id}_HHMMSS.tif")
    transform = from_origin(ulx, uly, xdim, ydim)
    
    with rasterio.open(
        out_path, 
        'w', 
        driver='GTiff', 
        height=rows, 
        width=cols, 
        count=1, 
        dtype='float64', 
        crs=crs_code, 
        transform=transform, 
        compress='lzw'
    ) as dst:
        dst.write(dn_values, 1)
        dst.update_tags(
            comment=f"Exact_Sensing_Time={mean_sensing_time.isoformat()}",
            parallax_correction="Applied (2.6s) with Platform Parity Logic",
            satellite=satellite,
            cloud_cover=str(cloud_cover),
            relative_orbit=str(relative_orbit)
        )
    
    print(f"  Success. Saved to {out_path}")
    
    return {
        "Scene_ID": scene_id,
        "Platform": satellite,
        "Relative_Orbit": relative_orbit,
        "Cloud_Cover_Percent": cloud_cover,
        "Exact_Mean_Sensing_Time": mean_sensing_time.isoformat()
    }

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    items = fetch_stac_items(START_DATE, END_DATE, TARGET_TILE, MAX_CLOUD_COVER)
    
    if items:
        processed_data = []
        for item in items:
            result = process_stac_item(item, OUTPUT_DIR)
            if result:
                processed_data.append(result)
        
        if processed_data:
            df = pd.DataFrame(processed_data)
            df = df.sort_values("Exact_Mean_Sensing_Time").reset_index(drop=True)
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"\nBatch processing complete. Summary saved to {OUTPUT_CSV}")
        else:
            print("\nNo scenes were successfully processed.")
    else:
        print("\nNo items found for the specified parameters. Execution halted.")
