import os, sys, gc
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Utilities.user_input import *
from Utilities.Functions.functions_data_processing import *
from Utilities.Functions.functions_plot import *
from Utilities.Functions.functions_statistics import *
from import_lib import *

# -----------------------------
# USER SETTINGS
# -----------------------------
folder_path_MODIS = r"D:\Data\MODIS\MYD06_L2_6.1-20251127_132549"
h5_path = r"C:\Users\andreap\Documents\Cloud_Effective_Radius_Bias_Correction_Algorithm\Bias_Correction_Algorithm\Data_Processing\Processed_Data\Small\MODIS_data.h5"

print("Starting MODIS processing including cloud coverage...")

# -----------------------------
# CREATE OUTPUT HDF5
# -----------------------------
os.makedirs(os.path.dirname(h5_path), exist_ok=True)

with h5py.File(h5_path, "w") as hf:

    # Create extendable datasets
    dsets = {}
    for key in ["radius", "cot", "sza", "lat", "lon", "cloud_coverage"]:
        dsets[key] = hf.create_dataset(
            key, shape=(0,), maxshape=(None,), dtype="float32", compression="gzip"
        )

    dsets["date"] = hf.create_dataset(
        "date", shape=(0,), maxshape=(None,), dtype="int64", compression="gzip"
    )

    all_files = [f for f in os.listdir(folder_path_MODIS) if f.endswith(".hdf")]
    n_total = len(all_files)
    n_runs = 0

    print(f"Found {n_total} MODIS files.")

    # -----------------------------------------
    # DEFINE CLOUD COVERAGE GRID
    # -----------------------------------------
    box_size = 0.5
    lat_bins = np.arange(-90, 90 + box_size, box_size)
    lon_bins = np.arange(-180, 180 + box_size, box_size)

    # -----------------------------------------
    # FILE LOOP
    # -----------------------------------------
    for filename in all_files:
        file_path = os.path.join(folder_path_MODIS, filename)
        n_runs += 1

        if n_runs % 50 == 0:
            print(f"Processed {n_runs}/{n_total} files...")

        # -----------------------------
        # READ MODIS HDF4 FILE
        # -----------------------------
        try:
            hdf = SD(file_path, SDC.READ)
        except Exception as e:
            print(f"ERROR reading file — Skipping: {file_path}")
            print("   Reason:", e)
            continue

        # Extract variables
        try:
            radius = hdf.select("Cloud_Effective_Radius_16")[:].astype(float) * 0.01
            cot = hdf.select("Cloud_Optical_Thickness_16")[:].astype(float) * 0.01
            sza = hdf.select("Solar_Zenith")[:].astype(float) * 0.01

            lat = hdf.select("Latitude")[:]
            lon = hdf.select("Longitude")[:]

            cloud_phase = hdf.select("Cloud_Phase_Optical_Properties")[:].astype(int)
            # water cloud = 2 → consistent with MODIS documentation
            cloud_flag = (cloud_phase == 2).astype(int)

        except Exception as e:
            print(f"Variable missing — skipping file: {filename}")
            print("   Reason:", e)
            continue

        # ----------------------------------------------
        # UPSAMPLE SZA (5 km → 1 km grid)
        # ----------------------------------------------
        if sza.shape != radius.shape:

            factor_row = radius.shape[0] // sza.shape[0]
            factor_col = radius.shape[1] // sza.shape[1]

            if factor_row < 1 or factor_col < 1:
                print("SZA grid unexpectedly larger than radius grid, skipping:", filename)
                continue

            sza = np.repeat(np.repeat(sza, factor_row, axis=0), factor_col, axis=1)
            sza = sza[:radius.shape[0], :radius.shape[1]]

        # ----------------------------------------------
        # GRID MATCHING
        # ----------------------------------------------
        min_rows = min(
            radius.shape[0], cot.shape[0], sza.shape[0],
            lat.shape[0], lon.shape[0], cloud_flag.shape[0]
        )
        min_cols = min(
            radius.shape[1], cot.shape[1], sza.shape[1],
            lat.shape[1], lon.shape[1], cloud_flag.shape[1]
        )

        radius      = radius[:min_rows, :min_cols]
        cot         = cot[:min_rows, :min_cols]
        sza         = sza[:min_rows, :min_cols]
        lat         = lat[:min_rows, :min_cols]
        lon         = lon[:min_rows, :min_cols]
        cloud_flag  = cloud_flag[:min_rows, :min_cols]

        # ----------------------------------------------
        # FILTER INVALID PIXELS
        # ----------------------------------------------
        mask = np.zeros_like(radius, dtype=bool)

        mask |= (radius < 0) | np.isnan(radius)
        mask |= (cot < 0) | np.isnan(cot)
        mask |= (sza < 0) | np.isnan(sza)

        if not world_data:
            mask |= ~(
                (lat >= lat_min) & (lat <= lat_max) &
                (lon >= lon_min) & (lon <= lon_max)
            )

        # remove longitude discontinuity
        lon_diff = np.abs(np.diff(lon, axis=1))
        discontinuity = lon_diff > 180
        discontinuity = np.hstack([
            np.zeros((discontinuity.shape[0], 1), dtype=bool),
            discontinuity
        ])
        mask |= discontinuity

        valid = ~mask

        # ----------------------------------------------
        # COMPUTE CLOUD COVERAGE (same method as OCI)
        # ----------------------------------------------
        lat_valid = lat[valid]
        lon_valid = lon[valid]
        cloud_valid = cloud_flag[valid]

        cloud_count, _, _ = np.histogram2d(lat_valid, lon_valid,
                                           bins=[lat_bins, lon_bins],
                                           weights=cloud_valid)

        pixel_count, _, _ = np.histogram2d(lat_valid, lon_valid,
                                           bins=[lat_bins, lon_bins])

        cloud_coverage_box = np.zeros_like(cloud_count)
        cloud_coverage_box[pixel_count > 0] = cloud_count[pixel_count > 0] / pixel_count[pixel_count > 0]

        lat_idx = np.clip(np.digitize(lat_valid, lat_bins) - 1, 0, cloud_coverage_box.shape[0]-1)
        lon_idx = np.clip(np.digitize(lon_valid, lon_bins) - 1, 0, cloud_coverage_box.shape[1]-1)

        cloud_coverage_pixel = cloud_coverage_box[lat_idx, lon_idx].astype(np.float32)
        # ----------------------------------------------
        # SAVE VARIABLES
        # ----------------------------------------------
        for key, arr in zip(
            ["radius", "cot", "sza", "lat", "lon", "cloud_coverage"],
            [radius, cot, sza, lat, lon, cloud_coverage_pixel]
        ):

            valid_values = (
                arr[valid].ravel().astype(np.float32)
                if key != "cloud_coverage"
                else cloud_coverage_pixel
            )

            prev = dsets[key].shape[0]
            dsets[key].resize(prev + len(valid_values), axis=0)
            dsets[key][prev:] = valid_values

        # ----------------------------------------------
        # SAVE DATE
        # ----------------------------------------------
        try:
            # You probably want to extract this from filename, adjust here
            date_int = 20250807
            prev = dsets["date"].shape[0]
            dsets["date"].resize(prev + 1, axis=0)
            dsets["date"][prev] = date_int
        except:
            pass

        gc.collect()

print(f"\nFinished processing {n_runs}/{n_total} MODIS files.")
print(f"Saved processed MODIS cloud data (incl. cloud coverage) to: {h5_path}")
