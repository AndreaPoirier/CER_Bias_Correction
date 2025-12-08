import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from import_lib import *


def is_in_region(lat_granular, lon_granular, lat_min, lat_max, lon_min, lon_max):
    '''
    Checks if any part of the granular data overlaps with the region of interest. If it does, function returns True.

    input
        lat_granular lon_granular -->  array containing the coordinates for each data point for 1 granular
        lat_min, lat_max, lon_min, lon_max --> float of the region of interest (rectangle shape)

    output
        lat_overlap, lon_overlap --> boolean 
    '''
    
    lat_overlap = np.any((lat_granular >= lat_min) & (lat_granular <= lat_max))
    lon_overlap = np.any((lon_granular >= lon_min) & (lon_granular <= lon_max))
    return lat_overlap and lon_overlap



def filter_region(lat, lon, val,lat_min,lon_min,lat_max,lon_max):
    mask = (
        (lat >= lat_min) & (lat <= lat_max) &
        (lon >= lon_min) & (lon <= lon_max)
    )
    return lat[mask], lon[mask], val[mask]




def is_within_n_days(filename, n, start_day):
    """
    Check if the file's date is within the first n days from the initial day.

    Parameters
    ----------
    filename : str
        Filename containing date in format YYYYMMDD, e.g., 'PACE_HARP2.20250801T001440.L2.CLOUD_GPC.V3_0'
    n : int
        Number of days to include from the initial day.
    start_day : str or datetime
        Initial day in 'YYYYMMDD' format or a datetime object.

    Returns
    -------
    bool
        True if file's date is within the first n days, False otherwise.
    """

    # Convert start_day to datetime if it's a string
    if isinstance(start_day, str):
        start_day = datetime.strptime(start_day, "%Y%m%d")

    # Extract date from filename
    try:
        file_date_str = filename.split('.')[1].split('T')[0]  # '20250801'
        file_date = datetime.strptime(file_date_str, "%Y%m%d")
    except Exception:
        # If filename format is unexpected, return False
        return False

    # Check if file date is within n days
    return start_day <= file_date < start_day + timedelta(days=n)


def aggregate_oci_to_harp2_radius(lat_oci, lon_oci, rad_oci, lat_harp, lon_harp, radius_km=15.0):
    """
    Aggregate OCI measurements within a radius around each HARP2 point.
    Returns mean values aligned with HARP2.
    """
    def deg_to_km_latlon(lat):
        return 111.0, 111.0 * np.cos(np.deg2rad(lat))
    
    # Convert radius in km to degrees using average latitude
    mean_lat = np.nanmean(lat_harp)
    km_per_deg_lat, km_per_deg_lon = deg_to_km_latlon(mean_lat)
    avg_km_per_deg = (km_per_deg_lat + km_per_deg_lon) / 2.0
    radius_deg = radius_km / avg_km_per_deg

    # Build KDTree for OCI points
    oci_points = np.column_stack((lat_oci, lon_oci))
    tree = cKDTree(oci_points)

    harp_points = np.column_stack((lat_harp, lon_harp))
    idx_list = tree.query_ball_point(harp_points, r=radius_deg)

    agg_vals = np.full(len(lat_harp), np.nan)
    counts = np.zeros(len(lat_harp), dtype=int)
    for i, inds in enumerate(idx_list):
        if len(inds) > 0:
            vals = rad_oci[inds]
            vals = vals[~np.isnan(vals)]
            if vals.size > 0:
                agg_vals[i] = np.mean(vals)  # or np.median(vals)
                counts[i] = vals.size
    valid_mask = ~np.isnan(agg_vals)
    return agg_vals, counts, valid_mask

def print_pair_stats(y_ref, y_model, label):
    rmse = np.sqrt(mean_squared_error(y_ref, y_model))
    mae = mean_absolute_error(y_ref, y_model)
    bias = np.mean(y_model - y_ref)
    eps = 1e-8
    aare = np.mean(np.abs((y_model - y_ref) / (y_ref + eps))) * 100.0
    error_array = np.abs(y_model - y_ref)
    print(f"--- {label} ---")
    print(f"Count: {len(y_ref)}")
    print(f"RMSE: {rmse:.6f}  Mean Absolute Error: {mae:.6f}  Bias: {bias:.6f}")
    print(f"Absolute Relative Error: {aare:.3f}%  ")
    return len(y_ref), rmse, mae, bias, aare, error_array

def log_metrics(
    label,
    count,
    rmse,
    mae,
    bias,
    aare,
    output_file,
    mode,
    results_sorted,
    top_n=10
):
    """Logs evaluation metrics and model ranking results to a file and prints them."""
    with open(output_file, mode) as f:
        f.write("\n=== TOP VARIABLE COMBINATIONS ===\n")
        f.write(f"Showing top {min(top_n, len(results_sorted))} combinations:\n")

        for i, (names, r2, mae_val) in enumerate(results_sorted[:top_n], 1):
            f.write(f"{i}. Combo: {names}: Test MAE SF = {mae_val:.4f}, Test R2 = {r2:.4f}\n")

        # Best model (lowest MAE)
        best_names, best_r2, best_mae = results_sorted[0]
        f.write("\n=== BEST VARIABLE COMBINATION ===\n")
        f.write(f"Combo: {best_names}\n")
        f.write(f"Best SF Test MAE: {best_mae:.4f}\n")
        f.write(f"Corresponding R2: {best_r2:.4f}\n\n")

        f.write("=== PERFORMANCE METRICS FOR BEST COMBINATION===\n")

        for i, lbl in enumerate(label):
            f.write(f"\n--- {lbl} ---\n")
            f.write(f"Count: {count[i] if i < len(count) else count[-1]}\n")
            f.write(f"RMSE: {rmse[i]:.6f}\n")
            f.write(f"Mean Absolute Error: {mae[i]:.6f}\n")
            f.write(f"Bias: {bias[i]:.6f}\n")
            f.write(f"Absolute Relative Error: {aare[i]:.3f}%\n")

        

def is_corrupted(path):
    # Check HDF5 signature
    try:
        with h5py.File(path, "r"):
            pass
    except Exception:
        return True  # corrupted

    # Check NetCDF structure
    try:
        with Dataset(path, "r"):
            pass
    except Exception:
        return True  # corrupted

    return False  # OK


def removing_corrupted_files(base_folder):
    corrupted_count = 0
    total_files = []

    # Collect all .nc files
    for root, dirs, files in os.walk(base_folder):
        for filename in files:
            if filename.endswith(".nc"):
                total_files.append(os.path.join(root, filename))

    print(f"Found {len(total_files)} .nc files. Starting corruption check...\n")

    # tqdm progress bar
    for full_path in tqdm(total_files, desc="Checking for corrupted files", unit="file"):
        if is_corrupted(full_path):
            corrupted_count += 1
            os.remove(full_path)

    print("\n---------------------------")
    print("Total .nc files scanned:", len(total_files))
    print("Corrupted files removed:", corrupted_count)
    print("---------------------------")

def starting_code():
    user_input = input("Enter y to run file: ")
    if user_input.lower() != 'y':
        print("Stopping the program.")
        sys.exit() 
    print("Code is running...")