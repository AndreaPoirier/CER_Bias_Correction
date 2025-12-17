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
h5_path = os.path.join(folder_path_saved_processed_data, "OCI_data.h5")
variables_to_save = ["cer_16","cer_21", "cer_22","cot", "sza", "lat", "lon", "cth","cwp","cloud_coverage", "chi"]
all_files = [f for f in os.listdir(folder_path_OCI) if f.endswith(".nc")]
n_total = len(all_files)

# Printing of parameters
print("==== GETTING OCI DATA LOCALLY ====")
description_of_script = """This script takes the locally saved OCI .nc files and processes such that it can be used in get_processed_data.py. The output is the file OCI_data.h5.
"""
print(textwrap.fill(description_of_script, width=100))
print("========================================================================")
print(f"Loading data from {folder_path_OCI}")
print(f"Saving OCI data to {folder_path_saved_processed_data}\\OCI_data.h5")
print("========================================================================")
print(f"Using (sampling) {sample_fraction_files * 100} % of total OCI files")
print(f"Number of files {n_total}")
print("========================================================================")
if world_data:
    print("Getting data for the world")
else:
    print(f"Region box: lat_min: {lat_min:.2f} | lat_max: {lat_max:.2f} | lon_min: {lon_min:.2f} | lon_max: {lon_max:.2f}")
if day_data:
    print(f"Filtering files based on day")
else:
    print(f"Not filtering files based on day")
print("========================================================================")
print(f"Variables saved {variables_to_save}")
print("========================================================================")

# Giving option to user to run or not the code
starting_code()



######################
#### MAIN LOOP #######
######################

with h5py.File(h5_path, "w") as hf:
    dsets = {}
    for key in variables_to_save:
        dsets[key] = hf.create_dataset(
            key, shape=(0,), maxshape=(None,), dtype="float32", compression="gzip"
        )

    dsets["date"] = hf.create_dataset(
        "date", shape=(0,), maxshape=(None,), dtype="int64", compression="gzip"
    )

    ###########################
    #### FILE SAMPLING ########
    ###########################


    if sample_fraction_files == 1:
        sampled_files = all_files
        print("Using all files (no sampling).")
        n_sample = n_total
    else:
        random.seed(42)
        n_sample = max(1, int(n_total * sample_fraction_files))
        sampled_files = random.sample(all_files, k=n_sample)
        print(f"Using {n_sample} of {n_total} files ({n_sample/n_total:.1%})")


    ###########################
    #### FILE ITERATION #######
    ###########################

    n_runs = 0
    for filename in sampled_files:

        # Removing files not at correct date (if day_data
        if day_data and not is_within_n_days(filename, n_days, start_day): 
            continue
        
        # Printing progress
        file_path = os.path.join(folder_path_OCI, filename)
        n_runs += 1
        if n_runs % 50 == 0:
            print(f"Processed {n_runs}/{n_sample} files...")

        # Getting file date
        basename = filename.split(".")[1]
        file_date = int(basename[:8])

        # Opening file
        with Dataset(file_path, mode="r") as nc:

            # Access groups
            geo_physical_grp = nc.groups["geophysical_data"]
            geo_loc_grp = nc.groups["navigation_data"]
            scan_grp = nc.groups["scan_line_attributes"]

            # Extract variables
            cer_16 = geo_physical_grp.variables["cer_16"][:].astype(float)
            cer_21 = geo_physical_grp.variables["cer_21"][:].astype(float)
            cer_22 = geo_physical_grp.variables["cer_22"][:].astype(float)
            cot = geo_physical_grp.variables["cot_16"][:].astype(float)
            cth = geo_physical_grp.variables["cth"][:].astype(float)
            cwp = geo_physical_grp.variables["cwp_16"][:].astype(float)
            lat = geo_loc_grp.variables["latitude"][:]
            lon = geo_loc_grp.variables["longitude"][:]
            cloud_flag = geo_physical_grp.variables["water_cloud"][:].astype(int)
            sza = scan_grp.variables["csol_z"][:].astype(float)
            sza = sza[:, None] * np.ones_like(lon)

            # Mask invalid data (WITHOUT cloud_flag for coverage calculation)
            mask_for_coverage = np.zeros(lat.shape, dtype=bool)
            for var in [cer_16, cer_21, cer_22, cot, sza]:
                mask_for_coverage |= (var < 0) | np.isnan(var)

            # Mask longitude discontinuity
            lon_diff = np.abs(np.diff(lon, axis=1))
            discontinuity = lon_diff > 180
            discontinuity = np.hstack([np.zeros((discontinuity.shape[0], 1), dtype=bool), discontinuity])
            mask_for_coverage |= discontinuity
            mask_for_coverage |= (cer_16 < 0)

            # Masking data outside of region of interest (if not world_data)
            if not world_data:
                mask_for_coverage |= ~(
                    (lat >= lat_min) & (lat <= lat_max) &
                    (lon >= lon_min) & (lon <= lon_max)
                )

            valid_for_coverage = ~mask_for_coverage

            # Extract ALL valid pixels (including cloud_flag==0) for coverage calculation
            lat_all = lat[valid_for_coverage]
            lon_all = lon[valid_for_coverage]
            cloud_flag_all = cloud_flag[valid_for_coverage]

            # Build KDTree with ALL valid pixels
            coords_all = np.column_stack([lat_all, lon_all])
            tree_all = cKDTree(coords_all)

            # NOW apply cloud_flag mask for final data (cloudy pixels only)
            mask = mask_for_coverage.copy()
            mask |= (cloud_flag == 0)
            valid = ~mask

            # Extract valid values for final data (cloud_flag==0 removed)
            lat_v = lat[valid]
            lon_v = lon[valid]
            cot_v = cot[valid]

            # Computing chi and cloud coverage with sliding window
            coords = np.column_stack([lat_v, lon_v])
            chi = np.zeros(len(lat_v), dtype=np.float32)
            cloud_cov = np.zeros(len(lat_v), dtype=np.float32)

            for i, (la, lo) in enumerate(coords):
                # Find neighbors in ALL pixels (cloudy + non-cloudy) for cloud coverage
                idxs_all = tree_all.query_ball_point(
                    [la, lo],
                    r=box_width_for_chi_and_SF_computation,
                    p=np.inf
                )
                
                # Cloud coverage = mean of cloud_flag (1s and 0s)
                if len(idxs_all) > 0:
                    cloud_cov[i] = np.mean(cloud_flag_all[idxs_all])
                
                # For chi, we need neighbors from cloudy pixels only
                # Find which of the valid cloudy pixels are in the box
                distances = np.sqrt((lat_v - la)**2 + (lon_v - lo)**2)
                idxs_cloudy = np.where(distances <= box_width_for_chi_and_SF_computation)[0]
                
                if len(idxs_cloudy) > 0:
                    chi[i] = compute_chi(cot_v[idxs_cloudy])

            # Save variables to .h5 file
            for key, arr in zip(
                ["cer_16", "cer_21", "cer_22", "cot", "sza",
                "lat", "lon", "cth", "cwp",
                "cloud_coverage", "chi"],

                [cer_16, cer_21, cer_22, cot, sza,
                lat, lon, cth, cwp,
                cloud_cov, chi]
            ):
                valid_values = arr[valid].ravel().astype(np.float32) if key not in ["cloud_coverage", "chi"] else arr
                prev = dsets[key].shape[0]
                dsets[key].resize(prev + len(valid_values), axis=0)
                dsets[key][prev:] = valid_values

            # Save date (one per pixel)
            prev_date = dsets["date"].shape[0]
            dsets["date"].resize(prev_date + len(lat_v), axis=0)
            dsets["date"][prev_date:] = file_date

        gc.collect()

print(f"Finished processing {n_runs}/{n_sample} files.")
print(f"Data saved incrementally to: {h5_path}")


