import os, sys, random, gc
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utilities.user_input import *
from utilities.processing import *
from utilities.plotting import *
from utilities.statistics import *
from import_lib import *

###########################
#### USER INTERFACE #######
###########################

# File path, variables to save, and file count
h5_path = os.path.join(folder_path_saved_processed_data, "MODIS_data.h5")
variables_to_save = ["radius", "cot", "sza", "lat", "lon", "cloud_coverage"]
all_files = [f for f in os.listdir(folder_path_MODIS) if f.endswith(".hdf")]
n_total = len(all_files)

# Printing of run parameters 
print("==== GETTING MODIS DATA LOCALLY ====")
print(f"Loading data from {folder_path_MODIS}")
print(f"Saving MODIS data to {folder_path_saved_processed_data}\MODIS_data.h5")
print("========================================================================")
print(f"Using (sampling) {sample_fraction_files * 100} % of total MODIS files")
print(f"Number of files {n_total}")
print("========================================================================")
if world_data:
    print("Getting data for the world")
else:
    print(f"Region box: "f"lat_min: {lat_min:.2f} | "f"lat_max: {lat_max:.2f} | "f"lon_min: {lon_min:.2f} | "f"lon_max: {lon_max:.2f}")
print(f"Variables saved {variables_to_save}")
print("========================================================================")

# Giving option to user to run or not the code
starting_code()


######################
#### MAIN LOOP #######
######################

os.makedirs(os.path.dirname(h5_path), exist_ok=True)

# Opening .h5 file where results will be saved
with h5py.File(h5_path, "w") as hf:

    # Create dataset for each variable to be saved
    dsets = {}
    for key in ["radius", "cot", "sza", "lat", "lon", "cloud_coverage"]:
        dsets[key] = hf.create_dataset(
            key, shape=(0,), maxshape=(None,), dtype="float32", compression="gzip"
        )

    # Create date dataset
    dsets["date"] = hf.create_dataset(
        "date", shape=(0,), maxshape=(None,), dtype="int64", compression="gzip"
    )

    all_files = [f for f in os.listdir(folder_path_MODIS) if f.endswith(".hdf")]
    n_total = len(all_files)
    n_runs = 0

    ###########################
    #### FILE SAMPLING ########
    ###########################
    
    if sample_fraction_files == 1:
        print("Using all files (no sampling).")
    else:
        random.seed(42)
        n_sample = max(1, int(n_total * sample_fraction_files))
        all_files = random.sample(all_files, k=n_sample)
        print(f"Using {n_sample} of {n_total} files ({n_sample/n_total:.1%})")

    # Define cloud coverage grid
    box_size = 0.5
    lat_bins = np.arange(-90, 90 + box_size, box_size)
    lon_bins = np.arange(-180, 180 + box_size, box_size)

    ###########################
    #### FILE ITERATION #######
    ###########################

    for filename in all_files:
        file_path = os.path.join(folder_path_MODIS, filename)

        # Printing progress
        n_runs += 1
        if n_runs % 50 == 0:
            print(f"Processed {n_runs}/{n_total} files...")

    
        # Trying to read .hdf4 MODIS files
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

            # water cloud = 2 
            cloud_flag = (cloud_phase == 2).astype(int)

        except Exception as e:
            print(f"Variable missing — skipping file: {filename}")
            print("   Reason:", e)
            continue

        # Upsampling SZA (5 km → 1 km grid)
        if sza.shape != radius.shape:

            factor_row = radius.shape[0] // sza.shape[0]
            factor_col = radius.shape[1] // sza.shape[1]

            if factor_row < 1 or factor_col < 1:
                print("SZA grid unexpectedly larger than radius grid, skipping:", filename)
                continue

            sza = np.repeat(np.repeat(sza, factor_row, axis=0), factor_col, axis=1)
            sza = sza[:radius.shape[0], :radius.shape[1]]

        # Grid matching
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

        # Filtering invalid pixels
        mask = np.zeros_like(radius, dtype=bool)

        mask |= (radius < 0) | np.isnan(radius)
        mask |= (cot < 0) | np.isnan(cot)
        mask |= (sza < 0) | np.isnan(sza)

        # Masking data outside of region of interest (if not world_data)
        if not world_data:
            mask |= ~(
                (lat >= lat_min) & (lat <= lat_max) &
                (lon >= lon_min) & (lon <= lon_max)
            )

        # Mask longitude discontinuity
        lon_diff = np.abs(np.diff(lon, axis=1))
        discontinuity = lon_diff > 180
        discontinuity = np.hstack([
            np.zeros((discontinuity.shape[0], 1), dtype=bool),
            discontinuity
        ])
        mask |= discontinuity

        valid = ~mask & (cloud_flag == 1)

        # Compute cloud coverage
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
        
        # Save variables
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


        # Save date 
        try:
            date_int = 20250807 ### Change this to the date. Functionality to extract date from file name not implemented
            prev = dsets["date"].shape[0]
            dsets["date"].resize(prev + 1, axis=0)
            dsets["date"][prev] = date_int
        except:
            pass

        gc.collect()

print(f"\nFinished processing {n_runs}/{n_total} MODIS files.")
print(f"Saved processed MODIS cloud data to: {h5_path}")
