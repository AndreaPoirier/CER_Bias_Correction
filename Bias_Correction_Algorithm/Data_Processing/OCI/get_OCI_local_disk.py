# import os, sys, random
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from Utilities.user_input import *
# from Utilities.Functions.functions_plot import *
# from Utilities.Functions.functions_statistics import *
# from Utilities.Functions.functions_data_processing import *
# from import_lib import *


# print("Starting to download OCI data")
# # Create HDF5 file
# h5_path = os.path.join(folder_path_saved_processed_data, "OCI_data.h5")
# with h5py.File(h5_path, "w") as hf:
#     # Precreate datasets with extendable dimensions
#     dsets = {}
#     for key in ["radius", "cot", "sza", "lat", "lon"]:
#         dsets[key] = hf.create_dataset(
#             key, shape=(0,), maxshape=(None,), dtype="float32", compression="gzip"
#         )
#     # date dataset
#     dsets["date"] = hf.create_dataset(
#         "date", shape=(0,), maxshape=(None,), dtype="int64", compression="gzip"
#     )
#     ###########################
#     #### FILE SAMPLING ########
#     ###########################

#     all_files = [f for f in os.listdir(folder_path_OCI) if f.endswith(".nc")]
#     n_total = len(all_files)

#     if sample_fraction_files == 1:
#         sampled_files = all_files
#         print(f"Using all files (no sampling).")
#         n_sample = n_total
#     else:
#         random.seed(42)
#         n_sample = max(1, int(n_total * sample_fraction_files))
#         sampled_files = random.sample(all_files, k=n_sample)
#         print(f"Using (sampling) {n_sample} of {n_total} total OCI files ({n_sample/n_total:.1%})")

#     ###########################
#     #### FILE ITERATION #######
#     ###########################
#     n_runs = 0
#     for filename in sampled_files:

#         if day_data and not is_within_n_days(filename, n_days, start_day):
#             continue

#         file_path = os.path.join(folder_path_OCI, filename)
#         n_runs += 1
#         if n_runs % 50 == 0:
#             print(f"Processed {n_runs}/{n_sample} files...")

#         basename = filename.split(".")[1]
#         file_date = int(basename[:8])

#         with Dataset(file_path, mode="r") as nc:
#             geo_physical_grp = nc.groups["geophysical_data"]
#             geo_loc_grp = nc.groups["navigation_data"]
#             scan_grp = nc.groups["scan_line_attributes"]

#             radius = geo_physical_grp.variables["cer_16"][:].astype(float)
#             cot = geo_physical_grp.variables["cot_16"][:].astype(float)
#             lat = geo_loc_grp.variables["latitude"][:]
#             lon = geo_loc_grp.variables["longitude"][:]
#             cloud_flag = geo_physical_grp.variables["water_cloud"][:].astype(int)
#             sza = scan_grp.variables["csol_z"][:].astype(float)
#             sza = sza[:, None] * np.ones_like(lon)
            
#             if not world_data:
#                 inside_any = np.any(
#                     (lat >= lat_min) & (lat <= lat_max) &
#                     (lon >= lon_min) & (lon <= lon_max)
#                 )
#                 if not inside_any:
#                     continue

#             mask = np.zeros(lat.shape, dtype=bool)

#             for var in [radius, cot, sza]:
#                 mask |= (var < 0) | np.isnan(var)

#             if not world_data:
#                 mask |= ~(
#                     (lat >= lat_min) & (lat <= lat_max) &
#                     (lon >= lon_min) & (lon <= lon_max)
#                 )

#             lon_diff = np.abs(np.diff(lon, axis=1))
#             discontinuity = lon_diff > 180
#             discontinuity = np.hstack([np.zeros((discontinuity.shape[0], 1), dtype=bool), discontinuity])
#             mask |= discontinuity
#             mask |= (cloud_flag == 0) | (radius < 0)

#             valid = ~mask

#             for key, arr in zip(["radius", "cot", "sza", "lat", "lon"], [radius, cot, sza, lat, lon]):
#                 valid_values = arr[valid].ravel().astype(np.float32)

#                 if len(valid_values) > 0:
#                     prev_size = dsets[key].shape[0]
#                     dsets[key].resize(prev_size + len(valid_values), axis=0)
#                     dsets[key][prev_size:] = valid_values

#             prev_size_date = dsets["date"].shape[0]
#             dsets["date"].resize(prev_size_date + len(valid_values), axis=0)
#             dsets["date"][prev_size_date:] = file_date

#             pixel_sizes = [5 * 360/(2*np.pi*6371*np.cos(np.deg2rad(np.mean(lat))))] 
            
#         gc.collect()

# print(f"Finished processing {n_runs}/{n_sample} files.")
# print(f"Data saved incrementally to: {h5_path}")

# Updated script computing cloud coverage around each pixel using cloud mask (1=cloud, 0=not cloud)

import os, sys, random, gc
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from Utilities.user_input import *
from Utilities.Functions.functions_data_processing import *
from Utilities.Functions.functions_plot import *
from Utilities.Functions.functions_statistics import *
from import_lib import *

print("Starting to download OCI data")

# Create HDF5 file
h5_path = os.path.join(folder_path_saved_processed_data, "OCI_data.h5")
with h5py.File(h5_path, "w") as hf:
    # Precreate datasets with extendable dimensions
    dsets = {}
    for key in ["radius", "cot", "sza", "lat", "lon", "cloud_coverage"]:
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
    all_files = [f for f in os.listdir(folder_path_OCI) if f.endswith(".nc")]
    n_total = len(all_files)

    if sample_fraction_files == 1:
        sampled_files = all_files
        print(f"Using all files (no sampling).")
        n_sample = n_total
    else:
        random.seed(42)
        n_sample = max(1, int(n_total * sample_fraction_files))
        sampled_files = random.sample(all_files, k=n_sample)
        print(f"Using (sampling) {n_sample} of {n_total} total OCI files ({n_sample/n_total:.1%})")

    ###########################
    #### FILE ITERATION #######
    ###########################
    n_runs = 0
    box_size = 0.5  # degrees for cloud coverage calculation

    # Define global bin grid once
    lat_bins = np.arange(-90, 90 + box_size, box_size)
    lon_bins = np.arange(-180, 180 + box_size, box_size)

    for filename in sampled_files:

        if day_data and not is_within_n_days(filename, n_days, start_day):
            continue

        file_path = os.path.join(folder_path_OCI, filename)
        n_runs += 1
        if n_runs % 50 == 0:
            print(f"Processed {n_runs}/{n_sample} files...")

        basename = filename.split(".")[1]
        file_date = int(basename[:8])

        with Dataset(file_path, mode="r") as nc:
            geo_physical_grp = nc.groups["geophysical_data"]
            geo_loc_grp = nc.groups["navigation_data"]
            scan_grp = nc.groups["scan_line_attributes"]

            radius = geo_physical_grp.variables["cer_16"][:].astype(float)
            cot = geo_physical_grp.variables["cot_16"][:].astype(float)
            lat = geo_loc_grp.variables["latitude"][:]
            lon = geo_loc_grp.variables["longitude"][:]
            cloud_flag = geo_physical_grp.variables["water_cloud"][:].astype(int)  # 1 = cloud, 0 = no cloud
            sza = scan_grp.variables["csol_z"][:].astype(float)
            sza = sza[:, None] * np.ones_like(lon)

            # Mask invalid pixels
            mask = np.zeros(lat.shape, dtype=bool)
            for var in [radius, cot, sza]:
                mask |= (var < 0) | np.isnan(var)

            # Longitude discontinuity mask
            lon_diff = np.abs(np.diff(lon, axis=1))
            discontinuity = lon_diff > 180
            discontinuity = np.hstack([np.zeros((discontinuity.shape[0], 1), dtype=bool), discontinuity])
            mask |= discontinuity

            # Also remove invalid cloud_flag or negative radius
            mask |= (cloud_flag < 0) | (radius < 0)

            # Geographic constraints
            if not world_data:
                mask |= ~(
                    (lat >= lat_min) & (lat <= lat_max) &
                    (lon >= lon_min) & (lon <= lon_max)
                )

            # Everything else valid
            valid = ~mask

            # Extract valid pixels
            lat_valid = lat[valid]
            lon_valid = lon[valid]
            cloud_valid = cloud_flag[valid]

            # Compute cloud coverage field in grid boxes
            cloud_count, _, _ = np.histogram2d(lat_valid, lon_valid, bins=[lat_bins, lon_bins], weights=cloud_valid)
            pixel_count, _, _ = np.histogram2d(lat_valid, lon_valid, bins=[lat_bins, lon_bins])

            cloud_coverage_box = np.zeros_like(cloud_count)
            cloud_coverage_box[pixel_count > 0] = cloud_count[pixel_count > 0] / pixel_count[pixel_count > 0]

            # Map box values back to each valid pixel
            lat_idx = np.clip(np.digitize(lat_valid, lat_bins) - 1, 0, cloud_coverage_box.shape[0]-1)
            lon_idx = np.clip(np.digitize(lon_valid, lon_bins) - 1, 0, cloud_coverage_box.shape[1]-1)
            cloud_coverage_pixel = cloud_coverage_box[lat_idx, lon_idx].astype(np.float32)

            valid_flat = valid.ravel()

            # Save pixel-level variables
            for key, arr in zip(["radius", "cot", "sza", "lat", "lon"],
                                [radius, cot, sza, lat, lon]):
                arr_flat = arr.ravel()
                valid_values = arr_flat[valid_flat].astype(np.float32)
                if len(valid_values) > 0:
                    prev = dsets[key].shape[0]
                    dsets[key].resize(prev + len(valid_values), axis=0)
                    dsets[key][prev:] = valid_values

            # Save cloud coverage
            prev = dsets["cloud_coverage"].shape[0]
            dsets["cloud_coverage"].resize(prev + len(cloud_coverage_pixel), axis=0)
            dsets["cloud_coverage"][prev:] = cloud_coverage_pixel
            # Save date
            prev_date = dsets["date"].shape[0]
            dsets["date"].resize(prev_date + len(cloud_coverage_pixel), axis=0)
            dsets["date"][prev_date:] = file_date

        gc.collect()

print(f"Finished processing {n_runs}/{n_sample} files.")
print(f"Data saved incrementally to: {h5_path}")

