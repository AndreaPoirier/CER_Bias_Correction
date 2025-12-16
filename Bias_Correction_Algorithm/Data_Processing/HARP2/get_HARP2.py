import os, sys, random, gc
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from Utilities.user_input import *
from Utilities.Functions.functions_data_processing import *
from Utilities.Functions.functions_plot import *
from Utilities.Functions.functions_statistics import *
from import_lib import *


###########################
#### USER INTERFACE #######
###########################


# File path, variables to save, and file count
h5_path = os.path.join(folder_path_saved_processed_data, "HARP2_data.h5")
all_files = [f for f in os.listdir(folder_path_HARP2) if f.endswith(".nc")]
n_total = len(all_files)
variables_to_save = ["radius", "cot", "lat", "lon"]

# Printing of parameters 
print("==== GETTING HARP2 DATA LOCALLY ====")
description_of_script = """This script takes the locally saved HARP2 .nc files and processes such that it can be used in get_processed_data.py. The output is the file HARP2_data.h5.
"""
print(textwrap.fill(description_of_script, width=100))
print("========================================================================")
print(f"Loading data from {folder_path_HARP2}")
print(f"Saving HARP2 data to {folder_path_saved_processed_data}\HARP2_data.h5")
print("========================================================================")
print(f"Using (sampling) {sample_fraction_files * 100} % of total HARP2 files")
print(f"Number of files {n_total}")
print("========================================================================")
if world_data:
    print("Getting data for the world")
else:
    print(f"Region box: "f"lat_min: {lat_min:.2f} | "f"lat_max: {lat_max:.2f} | "f"lon_min: {lon_min:.2f} | "f"lon_max: {lon_max:.2f}")
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
    # date dataset
    dsets["date"] = hf.create_dataset(
        "date", shape=(0,), maxshape=(None,), dtype="int64", compression="gzip"
    )

    ###########################
    #### FILE SAMPLING ########
    ###########################


    if sample_fraction_files == 1:
        sampled_files = all_files
        print(f"Using all files (no sampling).")
        n_sample = n_total
    else:
        random.seed(42)
        n_sample = max(1, int(len(os.listdir(folder_path_OCI)) * sample_fraction_files))
        sampled_files = random.sample(all_files, k=n_sample)
        print(f"Using (sampling) {n_sample} of {n_total} total HARP2 files ({n_sample/n_total:.1%})")


    ###########################
    #### FILE ITERATION #######
    ###########################

    n_runs = 0
    for filename in sampled_files:

        # Removing files not at correct date (if day_data
        if day_data and not is_within_n_days(filename, n_days, start_day): 
            continue
        
        # Printing progress
        file_path = os.path.join(folder_path_HARP2, filename)
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
            geo_loc_grp = nc.groups["geolocation_data"]

            # Extract variables
            radius = geo_physical_grp.variables["cloud_bow_droplet_effective_radius"][:].astype(float)
            cot = geo_physical_grp.variables["cloud_optical_thickness"][:].astype(float)
            lat = geo_loc_grp.variables["latitude"][:]
            lon = geo_loc_grp.variables["longitude"][:]

            # Mask invalid data
            mask = np.zeros(lat.shape, dtype=bool)
            mask |= np.isnan(radius) | np.isnan(cot)
            mask |= (radius <= 0) | (cot <= 0)

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
            mask |= (radius >= 30) | (radius < 0)

            valid = ~mask

            # Save variables to .h5 file
            for key, arr in zip(["radius", "cot", "lat", "lon"], [radius, cot, lat, lon]):
                valid_values = arr[valid].ravel().astype(np.float32)

                if len(valid_values) > 0:
                    prev_size = dsets[key].shape[0]
                    dsets[key].resize(prev_size + len(valid_values), axis=0)
                    dsets[key][prev_size:] = valid_values

            prev_size_date = dsets["date"].shape[0]
            dsets["date"].resize(prev_size_date + len(valid_values), axis=0)
            dsets["date"][prev_size_date:] = file_date
            
        gc.collect()

print(f"Finished processing {n_runs}/{n_sample} files.")
print(f"Data saved incrementally to: {h5_path}")