import os
from datetime import date
import numpy as np
import xarray as xr
import rioxarray
import rasterio
from rasterio.enums import Resampling
import cdsapi

# --------------------------------------------------
# User settings
# --------------------------------------------------
target_date = date(2026, 4, 5)
ndwi_anomaly_path = "ndwi_classic_zscore_doy_window.tif"

# AOI in CDS order: North, West, South, East
cds_area = [49.44, 15.55, 49.35, 15.69]

# Local afternoon hours in UTC approximation.
# Czech Republic in early April is CEST (UTC+2), so 13:00–16:00 local ~ 11:00–14:00 UTC.
afternoon_hours_utc = ["11:00", "12:00", "13:00", "14:00"]

# Initial thresholds
ndwi_fire_prone_thr = -1.0
ndwi_very_fire_prone_thr = -1.5
vpd_fire_prone_thr_kpa = 1.5
vpd_very_fire_prone_thr_kpa = 2.0

era5_nc = f"era5land_{target_date.isoformat()}.nc"
vpd_tif = f"vpd_afternoon_max_{target_date.isoformat()}.tif"
fire_tif = f"fire_proneness_{target_date.isoformat()}.tif"

# --------------------------------------------------
# 1. Download ERA5-Land hourly T and Td
# --------------------------------------------------
def download_era5land_hourly(target_date: date, area, out_nc: str):
    if os.path.exists(out_nc):
        print(f"Using existing {out_nc}")
        return out_nc

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-land",
        {
            "variable": [
                "2m_temperature",
                "2m_dewpoint_temperature",
            ],
            "year": f"{target_date.year}",
            "month": f"{target_date.month:02d}",
            "day": f"{target_date.day:02d}",
            "time": afternoon_hours_utc,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": area,
        },
        out_nc,
    )
    return out_nc

# --------------------------------------------------
# 2. Compute VPD from ERA5-Land
# --------------------------------------------------
def compute_vpd_kpa_from_ds(ds: xr.Dataset) -> xr.DataArray:
    """
    Uses:
      - t2m: 2m_temperature [K]
      - d2m: 2m_dewpoint_temperature [K]
    Returns VPD [kPa]
    """
    t_c = ds["t2m"] - 273.15
    td_c = ds["d2m"] - 273.15

    es = 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))
    ea = 0.6108 * np.exp((17.27 * td_c) / (td_c + 237.3))
    vpd = es - ea
    vpd.name = "vpd_kpa"
    return vpd

# --------------------------------------------------
# 3. Build same-day afternoon max VPD raster
# --------------------------------------------------
def build_vpd_raster(era5_nc_path: str, out_tif: str) -> xr.DataArray:
    ds = xr.open_dataset(era5_nc_path)

    # ERA5-Land NetCDF commonly uses latitude/longitude dims
    # Ensure CRS is set for rioxarray operations
    if "longitude" in ds.dims and "latitude" in ds.dims:
        ds = ds.rename({"longitude": "x", "latitude": "y"})

    vpd = compute_vpd_kpa_from_ds(ds)

    # Same-day afternoon maximum VPD
    vpd_max = vpd.max(dim="time")

    # Attach CRS and spatial dims
    vpd_max = vpd_max.rio.write_crs("EPSG:4326", inplace=False)
    vpd_max = vpd_max.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

    # Save coarse VPD raster
    vpd_max.rio.to_raster(out_tif)
    return vpd_max

# --------------------------------------------------
# 4. Match VPD to Sentinel-2 NDWI anomaly raster
# --------------------------------------------------
def reproject_vpd_to_ndwi(vpd_da: xr.DataArray, ndwi_path: str) -> xr.DataArray:
    ndwi = rioxarray.open_rasterio(ndwi_path).squeeze(drop=True)

    # Reproject coarse VPD to match NDWI raster grid
    # reproject_match is intended to align CRS, extent, and resolution
    vpd_matched = vpd_da.rio.reproject_match(
        ndwi,
        resampling=Resampling.bilinear
    )
    return vpd_matched

# --------------------------------------------------
# 5. Fire-proneness classification
# --------------------------------------------------
def classify_fire_proneness(ndwi_path: str, vpd_matched: xr.DataArray) -> xr.DataArray:
    ndwi = rioxarray.open_rasterio(ndwi_path).squeeze(drop=True)

    # Optional: mask non-burnable values if needed
    valid = np.isfinite(ndwi) & np.isfinite(vpd_matched)

    fire_prone = (
        (ndwi <= ndwi_fire_prone_thr) &
        (vpd_matched >= vpd_fire_prone_thr_kpa)
    )

    very_fire_prone = (
        (ndwi <= ndwi_very_fire_prone_thr) &
        (vpd_matched >= vpd_very_fire_prone_thr_kpa)
    )

    # Classes:
    # 0 = no data
    # 1 = not fire-prone
    # 2 = fire-prone
    # 3 = very fire-prone
    out = xr.full_like(ndwi, 0, dtype=np.uint8)
    out = xr.where(valid, 1, out)
    out = xr.where(valid & fire_prone, 2, out)
    out = xr.where(valid & very_fire_prone, 3, out)

    out.name = "fire_proneness"
    out.attrs["long_name"] = "Fire proneness from NDWI anomaly and VPD"
    out.attrs["classes"] = "0=no_data,1=not_fire_prone,2=fire_prone,3=very_fire_prone"

    # Keep NDWI raster georeferencing
    out = out.rio.write_crs(ndwi.rio.crs, inplace=False)
    return out

# --------------------------------------------------
# 6. Main
# --------------------------------------------------
if __name__ == "__main__":
    download_era5land_hourly(target_date, cds_area, era5_nc)
    print(f"Downloaded: {era5_nc}")

    vpd_da = build_vpd_raster(era5_nc, vpd_tif)
    print(f"Saved coarse VPD raster: {vpd_tif}")

    vpd_matched = reproject_vpd_to_ndwi(vpd_da, ndwi_anomaly_path)

    fire = classify_fire_proneness(ndwi_anomaly_path, vpd_matched)
    fire.rio.to_raster(fire_tif)
    print(f"Saved fire-proneness raster: {fire_tif}")
