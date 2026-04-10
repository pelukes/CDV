import openeo
from datetime import date, timedelta
from openeo.processes import absolute

# --------------------------------------------------
# 1. Authenticate with Copernicus Data Space Ecosystem
# --------------------------------------------------
conn = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

# --------------------------------------------------
# 2. Define AOI
# --------------------------------------------------
spatial_extent = {
    "west": 15.55,
    "south": 49.35,
    "east": 15.69,
    "north": 49.44,
    "crs": "EPSG:4326",
}

# --------------------------------------------------
# 3. Parameters
# --------------------------------------------------
target_date = date(2026, 4, 5)
window_half_size_days = 15
historical_years = range(2018, 2026)   # 2018-2025

# --------------------------------------------------
# 4. NDWI + SCL mask
# --------------------------------------------------
def get_masked_ndwi(cube):
    """
    NDWI/NDMI-like index from B8A and B11,
    masked to vegetation (SCL=4) and bare soil (SCL=5).
    """
    scl = cube.band("SCL")
    valid = (scl == 4) | (scl == 5)
    mask = ~valid

    b8a = cube.band("B8A")
    b11 = cube.band("B11")

    ndwi = (b8a - b11) / (b8a + b11)
    return ndwi.mask(mask)

# --------------------------------------------------
# 5. Build yearly DOY-centered intervals
# --------------------------------------------------
def build_yearly_intervals(center_month, center_day, half_window_days, years):
    """
    For each year create a right-open interval [start, end)
    around the same calendar day.
    """
    intervals = []
    for y in years:
        center = date(y, center_month, center_day)
        start = center - timedelta(days=half_window_days)
        end = center + timedelta(days=half_window_days + 1)
        intervals.append([start.isoformat(), end.isoformat()])
    return intervals

hist_intervals = build_yearly_intervals(
    center_month=target_date.month,
    center_day=target_date.day,
    half_window_days=window_half_size_days,
    years=historical_years
)

hist_start = hist_intervals[0][0]
hist_end = hist_intervals[-1][1]

# --------------------------------------------------
# 6. Load historical data
# --------------------------------------------------
cube_hist = conn.load_collection(
    "SENTINEL2_L2A",
    spatial_extent=spatial_extent,
    temporal_extent=[hist_start, hist_end],
    bands=["B8A", "B11", "SCL"]
)

ndwi_hist = get_masked_ndwi(cube_hist)

# Aggregate each historical year in its matching DOY window
yearly_ndwi = ndwi_hist.aggregate_temporal(
    intervals=hist_intervals,
    reducer="median"
)

# --------------------------------------------------
# 7. Historical baseline statistics
# --------------------------------------------------
# Classic statistics
baseline_mean = yearly_ndwi.reduce_dimension(dimension="t", reducer="mean")
baseline_sd = yearly_ndwi.reduce_dimension(dimension="t", reducer="sd")
baseline_sd = baseline_sd + 1e-6

# Robust statistics
baseline_median = yearly_ndwi.reduce_dimension(dimension="t", reducer="median")

deviation = yearly_ndwi - baseline_median
abs_dev = deviation.apply(absolute)

baseline_mad = abs_dev.reduce_dimension(dimension="t", reducer="median")

# MAD -> robust sigma approximation
robust_sigma = baseline_mad * 1.4826
robust_sigma = robust_sigma + 1e-6

# --------------------------------------------------
# 8. Load current-year data in the same DOY window
# --------------------------------------------------
current_start = (target_date - timedelta(days=window_half_size_days)).isoformat()
current_end = (target_date + timedelta(days=window_half_size_days + 1)).isoformat()

cube_current = conn.load_collection(
    "SENTINEL2_L2A",
    spatial_extent=spatial_extent,
    temporal_extent=[current_start, current_end],
    bands=["B8A", "B11", "SCL"]
)

current_ndwi = get_masked_ndwi(cube_current)
current_median = current_ndwi.reduce_dimension(dimension="t", reducer="median")

# --------------------------------------------------
# 9. Compute anomalies
# --------------------------------------------------
classic_z = (current_median - baseline_mean) / baseline_sd
robust_z = (current_median - baseline_median) / robust_sigma

# --------------------------------------------------
# 10. Export
# --------------------------------------------------
classic_z.download("ndwi_classic_zscore_doy_window.tif", format="GTiff")
print("Download complete: ndwi_classic_zscore_doy_window.tif")

robust_z.download("ndwi_robust_zscore_doy_window.tif", format="GTiff")
print("Download complete: ndwi_robust_zscore_doy_window.tif")
