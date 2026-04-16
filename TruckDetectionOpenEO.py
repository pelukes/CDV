import openeo
import xarray as xr
import rioxarray  # noqa: F401  # needed for .rio accessor
import numpy as np
import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from shapely.ops import substring
from rasterio import features
from rasterstats import zonal_stats
from s2cloudless import S2PixelCloudDetector


# ==========================================
# Configuration
# ==========================================
OPENEO_BACKEND = "https://openeofed.dataspace.copernicus.eu"

ROI = {"west": 16.5, "south": 49.1, "east": 16.7, "north": 49.3}
TIME_WINDOW = ["2025-04-01", "2025-08-31"]

TARGET_CRS = 32633
MAX_CLOUD_COVER = 20

BANDS_CLOUD = ["B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"]
BANDS_PARALLAX = ["B02", "B04", "SCL"]

RES_CLOUD = 60
RES_PARALLAX = 10

ROAD_BUFFER_M = 10
PARALLAX_THRESHOLD = 0.15
INVALID_SCL_CLASSES = [3, 8, 9, 10]

ROAD_CLASSES = [
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
]

S2CLOUDLESS_THRESHOLD = 0.4
S2CLOUDLESS_AVERAGE_OVER = 4
S2CLOUDLESS_DILATION = 2

FILE_60M = "s2_60m_clouds.nc"
FILE_10M = "s2_10m_parallax.nc"
FILE_OUT_NC = "s2_analyzed_results_10m.nc"
FILE_OUT_GEOJSON = "roads_100m_zonal_stats.geojson"


# ==========================================
# Utilities
# ==========================================
def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def connect_openeo():
    log("Connecting to OpenEO backend...")
    return openeo.connect(OPENEO_BACKEND).authenticate_oidc()


def attach_spatial_metadata_to_da(da: xr.DataArray, ref_ds: xr.Dataset) -> xr.DataArray:
    """
    Attach CRS, transform and coordinate-system metadata to a DataArray
    that contains x/y dimensions.
    """
    if "x" not in da.dims or "y" not in da.dims:
        return da

    ref_crs = ref_ds.rio.crs if ref_ds.rio.crs is not None else TARGET_CRS
    ref_transform = ref_ds.rio.transform()

    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    da = da.rio.write_crs(ref_crs, inplace=False)
    da = da.rio.write_transform(ref_transform, inplace=False)
    da = da.rio.write_coordinate_system(inplace=False)

    da.attrs.pop("grid_mapping", None)

    return da


def ensure_spatial_ref(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensure dataset and all x/y variables carry CRS and transform metadata.
    """
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

    ref_crs = ds.rio.crs if ds.rio.crs is not None else TARGET_CRS
    ds = ds.rio.write_crs(ref_crs, inplace=False)
    ds = ds.rio.write_coordinate_system(inplace=False)

    try:
        transform = ds.rio.transform()
        ds = ds.rio.write_transform(transform, inplace=False)
    except Exception:
        transform = None

    for var_name in ds.data_vars:
        da = ds[var_name]
        if "x" in da.dims and "y" in da.dims:
            da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
            da = da.rio.write_crs(ref_crs, inplace=False)
            da = da.rio.write_coordinate_system(inplace=False)
            if transform is not None:
                da = da.rio.write_transform(transform, inplace=False)
            da.attrs.pop("grid_mapping", None)
            ds[var_name] = da

    return ds


def strip_conflicting_cf_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Remove attributes that may conflict with xarray CF encoding during NetCDF export.
    """
    for var_name in ds.variables:
        ds[var_name].attrs.pop("grid_mapping", None)
    return ds


# ==========================================
# OpenEO download
# ==========================================
def download_openeo_data(conn) -> None:
    log(f"Downloading 60 m L1C dataset for s2cloudless (Max cloud: {MAX_CLOUD_COVER}%)...")
    cloud_filter = {"eo:cloud_cover": lambda v: v <= MAX_CLOUD_COVER}

    cube_60m = conn.load_collection(
        "SENTINEL2_L1C",
        spatial_extent=ROI,
        temporal_extent=TIME_WINDOW,
        bands=BANDS_CLOUD,
        properties=cloud_filter
    ).resample_spatial(resolution=RES_CLOUD, projection=TARGET_CRS)

    cube_60m.download(FILE_60M, format="NetCDF")

    log(f"Downloading 10 m L2A dataset for parallax analysis (Max cloud: {MAX_CLOUD_COVER}%)...")
    cube_10m = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=ROI,
        temporal_extent=TIME_WINDOW,
        bands=BANDS_PARALLAX,
        properties=cloud_filter
    ).resample_spatial(resolution=RES_PARALLAX, projection=TARGET_CRS)

    cube_10m.download(FILE_10M, format="NetCDF")


# ==========================================
# Dataset opening and alignment
# ==========================================
def open_dataset_checked(path: str) -> xr.Dataset:
    log(f"Opening dataset: {path}")
    ds = xr.open_dataset(path, decode_coords="all")

    if "t" not in ds.coords:
        raise ValueError(f"Dataset {path} does not contain time coordinate 't'.")
    if "x" not in ds.coords or "y" not in ds.coords:
        raise ValueError(f"Dataset {path} does not contain spatial coordinates 'x' and 'y'.")

    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

    if ds.rio.crs is None:
        ds = ds.rio.write_crs(TARGET_CRS, inplace=False)

    ds = ensure_spatial_ref(ds)
    return ds


def align_datasets_on_common_time(ds_60: xr.Dataset, ds_10: xr.Dataset):
    common_times = np.intersect1d(ds_60.t.values, ds_10.t.values)

    log(f"L1C scenes: {len(ds_60.t.values)}")
    log(f"L2A scenes: {len(ds_10.t.values)}")
    log(f"Common scenes: {len(common_times)}")

    if len(common_times) == 0:
        raise ValueError("No matching timestamps between 60 m L1C data and 10 m L2A data.")

    ds_60_aligned = ds_60.sel(t=common_times)
    ds_10_aligned = ds_10.sel(t=common_times)

    ds_60_aligned = ensure_spatial_ref(ds_60_aligned)
    ds_10_aligned = ensure_spatial_ref(ds_10_aligned)

    return ds_60_aligned, ds_10_aligned


# ==========================================
# s2cloudless
# ==========================================
def run_s2cloudless(ds_60: xr.Dataset):
    log("Running local s2cloudless on 60 m data...")

    for band in BANDS_CLOUD:
        if band not in ds_60.data_vars:
            raise ValueError(f"Missing required cloud band: {band}")

    stacked_data = np.stack([ds_60[band].values for band in BANDS_CLOUD], axis=-1)

    if stacked_data.ndim != 4:
        raise ValueError(f"Expected 4D array for s2cloudless input, got shape {stacked_data.shape}")
    if stacked_data.shape[-1] != len(BANDS_CLOUD):
        raise ValueError(
            f"Expected last dimension = {len(BANDS_CLOUD)}, got {stacked_data.shape[-1]}"
        )

    reflectances = np.clip(stacked_data.astype(np.float32) / 10000.0, 0.0, 1.0)

    detector = S2PixelCloudDetector(
        threshold=S2CLOUDLESS_THRESHOLD,
        average_over=S2CLOUDLESS_AVERAGE_OVER,
        dilation_size=S2CLOUDLESS_DILATION,
        all_bands=False,
    )

    cloud_mask_60m = detector.get_cloud_masks(reflectances).astype(np.uint8)
    cloud_probs_60m = detector.get_cloud_probability_maps(reflectances).astype(np.float32)

    mask_60m_da = xr.DataArray(
        cloud_mask_60m,
        coords={"t": ds_60.t, "y": ds_60.y, "x": ds_60.x},
        dims=("t", "y", "x"),
        name="s2cloudless_mask_60m",
    )

    probs_60m_da = xr.DataArray(
        cloud_probs_60m,
        coords={"t": ds_60.t, "y": ds_60.y, "x": ds_60.x},
        dims=("t", "y", "x"),
        name="s2cloudless_probability_60m",
    )

    mask_60m_da = attach_spatial_metadata_to_da(mask_60m_da, ds_60)
    probs_60m_da = attach_spatial_metadata_to_da(probs_60m_da, ds_60)

    return mask_60m_da, probs_60m_da


# ==========================================
# OSM roads and Zonal Stats
# ==========================================
def fetch_road_network() -> gpd.GeoDataFrame:
    log("Fetching OpenStreetMap road network...")

    ox.settings.timeout = 600
    ox.settings.log_console = True
    ox.settings.use_cache = True

    osm_bbox = box(ROI["west"], ROI["south"], ROI["east"], ROI["north"])

    try:
        roads_gdf = ox.features_from_polygon(osm_bbox, tags={"highway": ROAD_CLASSES})
    except AttributeError:
        roads_gdf = ox.geometries_from_polygon(osm_bbox, tags={"highway": ROAD_CLASSES})

    if roads_gdf.empty:
        raise ValueError("No road features found in ROI.")

    # Filter strictly for linear geometry
    roads_gdf = roads_gdf[roads_gdf.geometry.type.isin(["LineString", "MultiLineString"])]

    if roads_gdf.empty:
        raise ValueError("No linear road geometries found after geometry filtering.")

    # Project to metric system (UTM)
    roads_gdf = roads_gdf.to_crs(epsg=TARGET_CRS)
    return roads_gdf


def rasterize_roads_to_10m_grid(roads_gdf: gpd.GeoDataFrame, ds_10: xr.Dataset) -> np.ndarray:
    log("Rasterizing road buffer to match 10 m grid...")
    
    # Create temporary buffers just for the rasterization process
    buffered_roads = roads_gdf.geometry.buffer(ROAD_BUFFER_M)

    road_mask_2d = features.rasterize(
        shapes=((geom, 1) for geom in buffered_roads.values if geom is not None and not geom.is_empty),
        out_shape=(len(ds_10.y), len(ds_10.x)),
        transform=ds_10.rio.transform(),
        fill=0,
        dtype=np.uint8,
    )

    if road_mask_2d.sum() == 0:
        raise ValueError("Road rasterization produced an empty mask.")

    return road_mask_2d


def export_segmented_road_stats(
    roads_gdf: gpd.GeoDataFrame, 
    ds_10: xr.Dataset, 
    moving_targets: np.ndarray, 
    combined_cloud_mask: np.ndarray,
    file_out=FILE_OUT_GEOJSON
):
    log("Segmenting road network into 100m chunks...")
    
    segments = []
    # 1. Explode MultiLineStrings and segment into 100m chunks
    for _, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        elif geom.geom_type == "LineString":
            lines = [geom]
        else:
            continue
            
        for line in lines:
            dist = 0
            while dist < line.length:
                # Extract a 100m chunk along the line
                seg = substring(line, dist, dist + 100)
                segments.append({
                    "highway": row.get("highway", "unknown"),
                    "name": row.get("name", "unknown"),
                    "geometry": seg
                })
                dist += 100
                
    seg_gdf = gpd.GeoDataFrame(segments, crs=TARGET_CRS)
    log(f"Generated {len(seg_gdf)} individual 100m road segments.")
    
    # 2. OVERWRITE the geometry column with the buffer
    log(f"Converting line segments to {ROAD_BUFFER_M}m buffer polygons...")
    seg_gdf["geometry"] = seg_gdf.geometry.buffer(ROAD_BUFFER_M)
    
    # 3. Calculate Zonal Statistics for each timestamp using the polygon geometry
    log("Calculating zonal statistics for each observation day...")
    transform = ds_10.rio.transform()
    times = ds_10.t.values
    
    for i, t in enumerate(times):
        # Format the timestamp to YYYY-MM-DD for the attribute table column
        date_str = str(t)[:10] 
        col_det = f"det_{date_str}"
        col_qf = f"qf_{date_str}"
        
        # Array for detections
        frame_2d = moving_targets[i, :, :].astype(float)
        
        # Binary array for the quality flag: 1 where clear, 0 where clouded/shadowed
        # combined_cloud_mask: 0 is clear, 1 is invalid
        valid_frame_2d = (combined_cloud_mask[i, :, :] == 0).astype(float)
        
        # Calculate sum of anomalous pixels (Detections)
        stats_det = zonal_stats(
            seg_gdf.geometry, 
            frame_2d, 
            affine=transform, 
            stats="sum", 
            nodata=np.nan
        )
        
        # Calculate mean of valid pixels (Quality Flag Fraction)
        stats_qf = zonal_stats(
            seg_gdf.geometry, 
            valid_frame_2d, 
            affine=transform, 
            stats="mean", 
            nodata=np.nan
        )
        
        # Extract the metrics
        seg_gdf[col_det] = [s['sum'] if s['sum'] is not None else 0 for s in stats_det]
        seg_gdf[col_qf] = [round(s['mean'], 3) if s['mean'] is not None else 0.0 for s in stats_qf]

    # 4. Reproject to WGS84 (EPSG:4326) for standard GeoJSON compatibility
    log("Exporting buffered polygons to GeoJSON...")
    seg_gdf_4326 = seg_gdf.to_crs(epsg=4326)
    
    # Ensure columns are strings to prevent JSON serialization errors
    for col in seg_gdf_4326.columns:
        if col != "geometry" and seg_gdf_4326[col].dtype == object:
            seg_gdf_4326[col] = seg_gdf_4326[col].astype(str)

    seg_gdf_4326.to_file(file_out, driver="GeoJSON")
    log(f"GeoJSON export complete: {file_out}")


# ==========================================
# Grid alignment
# ==========================================
def upsample_cloud_mask_to_10m(mask_60m_da: xr.DataArray, probs_60m_da: xr.DataArray, ds_10: xr.Dataset):
    log("Upsampling 60 m cloud mask and probabilities to 10 m grid...")

    cloud_mask_10m = mask_60m_da.interp_like(ds_10, method="nearest")
    cloud_probs_10m = probs_60m_da.interp_like(ds_10, method="nearest")

    cloud_mask_10m = cloud_mask_10m.fillna(1).astype(np.uint8)
    cloud_probs_10m = cloud_probs_10m.astype(np.float32)

    cloud_mask_10m = attach_spatial_metadata_to_da(cloud_mask_10m, ds_10)
    cloud_probs_10m = attach_spatial_metadata_to_da(cloud_probs_10m, ds_10)

    return cloud_mask_10m, cloud_probs_10m


# ==========================================
# Parallax computation
# ==========================================
def compute_parallax_products(ds_10: xr.Dataset, cloud_mask_10m: xr.DataArray, road_mask_2d: np.ndarray):
    log("Computing combined mask and parallax anomalies...")

    for band in BANDS_PARALLAX:
        if band not in ds_10.data_vars:
            raise ValueError(f"Missing required parallax band: {band}")

    b02_10m = ds_10["B02"].astype(np.float32).values
    b04_10m = ds_10["B04"].astype(np.float32).values
    scl_10m = ds_10["SCL"].values

    parallax_index_raw = (b02_10m - b04_10m) / (b02_10m + b04_10m + 1e-8)

    scl_invalid_mask = np.isin(scl_10m, INVALID_SCL_CLASSES)
    combined_cloud_mask = ((cloud_mask_10m.values == 1) | scl_invalid_mask).astype(np.uint8)

    road_mask_3d = road_mask_2d[np.newaxis, :, :]
    valid_mask = (combined_cloud_mask == 0) & (road_mask_3d == 1)

    parallax_index_masked = np.where(valid_mask, parallax_index_raw, np.nan).astype(np.float32)
    moving_targets = (np.abs(parallax_index_masked) > PARALLAX_THRESHOLD).astype(np.uint8)

    return {
        "parallax_index_raw": parallax_index_raw.astype(np.float32),
        "combined_cloud_mask": combined_cloud_mask,
        "parallax_index_masked": parallax_index_masked,
        "moving_targets": moving_targets,
    }


# ==========================================
# Save output NetCDF
# ==========================================
def append_outputs_and_save(
    ds_10: xr.Dataset,
    cloud_mask_10m: xr.DataArray,
    cloud_probs_10m: xr.DataArray,
    road_mask_2d: np.ndarray,
    products: dict,
) -> None:
    log("Appending outputs and saving final NetCDF...")

    ds_out = ds_10.copy()
    ds_out = ensure_spatial_ref(ds_out)

    moving_targets_da = xr.DataArray(
        products["moving_targets"].astype(np.uint8),
        coords={"t": ds_10.t, "y": ds_10.y, "x": ds_10.x},
        dims=("t", "y", "x"),
        name="moving_targets",
    )

    combined_cloud_mask_da = xr.DataArray(
        products["combined_cloud_mask"].astype(np.uint8),
        coords={"t": ds_10.t, "y": ds_10.y, "x": ds_10.x},
        dims=("t", "y", "x"),
        name="combined_cloud_mask",
    )

    parallax_index_raw_da = xr.DataArray(
        products["parallax_index_raw"].astype(np.float32),
        coords={"t": ds_10.t, "y": ds_10.y, "x": ds_10.x},
        dims=("t", "y", "x"),
        name="parallax_index_raw",
    )

    parallax_index_masked_da = xr.DataArray(
        products["parallax_index_masked"].astype(np.float32),
        coords={"t": ds_10.t, "y": ds_10.y, "x": ds_10.x},
        dims=("t", "y", "x"),
        name="parallax_index_masked",
    )

    road_buffer_mask_da = xr.DataArray(
        road_mask_2d.astype(np.uint8),
        coords={"y": ds_10.y, "x": ds_10.x},
        dims=("y", "x"),
        name="road_buffer_mask",
    )

    moving_targets_da = attach_spatial_metadata_to_da(moving_targets_da, ds_10)
    combined_cloud_mask_da = attach_spatial_metadata_to_da(combined_cloud_mask_da, ds_10)
    parallax_index_raw_da = attach_spatial_metadata_to_da(parallax_index_raw_da, ds_10)
    parallax_index_masked_da = attach_spatial_metadata_to_da(parallax_index_masked_da, ds_10)
    road_buffer_mask_da = attach_spatial_metadata_to_da(road_buffer_mask_da, ds_10)

    cloud_mask_10m = attach_spatial_metadata_to_da(cloud_mask_10m, ds_10)
    cloud_probs_10m = attach_spatial_metadata_to_da(cloud_probs_10m, ds_10)

    ds_out["moving_targets"] = moving_targets_da
    ds_out["s2cloudless_mask"] = cloud_mask_10m
    ds_out["s2cloudless_probability"] = cloud_probs_10m
    ds_out["combined_cloud_mask"] = combined_cloud_mask_da
    ds_out["road_buffer_mask"] = road_buffer_mask_da
    ds_out["parallax_index_raw"] = parallax_index_raw_da
    ds_out["parallax_index_masked"] = parallax_index_masked_da

    ds_out = ensure_spatial_ref(ds_out)
    ds_out = strip_conflicting_cf_attrs(ds_out)

    ds_out.attrs["processing_backend"] = OPENEO_BACKEND
    ds_out.attrs["roi_west"] = ROI["west"]
    ds_out.attrs["roi_south"] = ROI["south"]
    ds_out.attrs["roi_east"] = ROI["east"]
    ds_out.attrs["roi_north"] = ROI["north"]
    ds_out.attrs["time_start"] = TIME_WINDOW[0]
    ds_out.attrs["time_end"] = TIME_WINDOW[1]
    ds_out.attrs["target_crs"] = TARGET_CRS
    ds_out.attrs["max_cloud_cover"] = MAX_CLOUD_COVER
    ds_out.attrs["road_buffer_m"] = ROAD_BUFFER_M
    ds_out.attrs["parallax_threshold"] = PARALLAX_THRESHOLD
    ds_out.attrs["invalid_scl_classes"] = ",".join(map(str, INVALID_SCL_CLASSES))
    ds_out.attrs["cloud_collection"] = "SENTINEL2_L1C"
    ds_out.attrs["parallax_collection"] = "SENTINEL2_L2A"

    encoding = {
        "moving_targets": {"zlib": True, "complevel": 4},
        "s2cloudless_mask": {"zlib": True, "complevel": 4},
        "s2cloudless_probability": {"zlib": True, "complevel": 4},
        "combined_cloud_mask": {"zlib": True, "complevel": 4},
        "road_buffer_mask": {"zlib": True, "complevel": 4},
        "parallax_index_raw": {"zlib": True, "complevel": 4},
        "parallax_index_masked": {"zlib": True, "complevel": 4},
    }

    ds_out.to_netcdf(FILE_OUT_NC, encoding=encoding)

    log("Spatial reference overview:")
    for var_name in ds_out.data_vars:
        if "x" in ds_out[var_name].dims and "y" in ds_out[var_name].dims:
            try:
                log(f"  {var_name}: CRS={ds_out[var_name].rio.crs}")
            except Exception:
                log(f"  {var_name}: CRS unavailable")

    log(f"Pipeline complete. Raster results saved to {FILE_OUT_NC}")


# ==========================================
# Main
# ==========================================
def main():
    conn = connect_openeo()

    download_openeo_data(conn)

    ds_60 = open_dataset_checked(FILE_60M)
    ds_10 = open_dataset_checked(FILE_10M)

    log(f"ds_60 CRS: {ds_60.rio.crs}")
    log(f"ds_10 CRS: {ds_10.rio.crs}")

    ds_60, ds_10 = align_datasets_on_common_time(ds_60, ds_10)

    mask_60m_da, probs_60m_da = run_s2cloudless(ds_60)

    roads_gdf = fetch_road_network()
    road_mask_2d = rasterize_roads_to_10m_grid(roads_gdf, ds_10)

    cloud_mask_10m, cloud_probs_10m = upsample_cloud_mask_to_10m(mask_60m_da, probs_60m_da, ds_10)
    
    products = compute_parallax_products(ds_10, cloud_mask_10m, road_mask_2d)

    append_outputs_and_save(
        ds_10=ds_10,
        cloud_mask_10m=cloud_mask_10m,
        cloud_probs_10m=cloud_probs_10m,
        road_mask_2d=road_mask_2d,
        products=products,
    )

    export_segmented_road_stats(
        roads_gdf=roads_gdf, 
        ds_10=ds_10, 
        moving_targets=products["moving_targets"],
        combined_cloud_mask=products["combined_cloud_mask"],
        file_out=FILE_OUT_GEOJSON
    )

if __name__ == "__main__":
    main()
