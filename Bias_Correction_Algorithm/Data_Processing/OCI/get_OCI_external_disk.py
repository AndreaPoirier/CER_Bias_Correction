import os, sys, gc
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from Utilities.user_input import *
from Utilities.Functions.functions_data_processing import *
from Utilities.Functions.functions_plot import *
from Utilities.Functions.functions_statistics import *
from import_lib import *

n_runs = 0
print("Starting to download OCI data")

# Create HDF5 file
h5_path = os.path.join(folder_path_saved_processed_data, "OCI_data.h5")
with h5py.File(h5_path, "w") as hf:
    # Precreate datasets with extendable dimensions
    dsets = {}
    for key in ["cer_16", "cer_21","cer_22","cot", "sza", "lat", "lon","cth","cwp","cloud_coverage"]:
        dsets[key] = hf.create_dataset(
            key, shape=(0,), maxshape=(None,), dtype="float32", compression="gzip"
        )
    # date dataset
    dsets["date"] = hf.create_dataset(
        "date", shape=(0,), maxshape=(None,), dtype="int64", compression="gzip"
    )

    years = ["2024", "2025"]
    months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
    days = ["01", "15"]
    base_folder = "D:\\Data\\OCI"

    box_size = 0.5  # degrees for cloud coverage calculation
    lat_bins = np.arange(-90, 90 + box_size, box_size)
    lon_bins = np.arange(-180, 180 + box_size, box_size)

    for year in years:
        for month in months:
            if (year == "2024" and month in ["01","02","03","04","05","06"]) or (year == "2025" and month in ["07","08","09","10","11","12"]):
                continue
            for day in days:
                folder_path = os.path.join(base_folder, year, month, day)
                all_files = [f for f in os.listdir(folder_path) if f.endswith(".nc")]
                n_runs += 1
                print(f"Processed {n_runs}/24 days...")
                print(f"Processing data from {year+month+day}\n")

                for filename in all_files:
                    file_path = os.path.join(folder_path, filename)
                    basename = filename.split(".")[1]
                    file_date = int(basename[:8])

                    with Dataset(file_path, mode="r") as nc:
                        geo_physical_grp = nc.groups["geophysical_data"]
                        geo_loc_grp = nc.groups["navigation_data"]
                        scan_grp = nc.groups["scan_line_attributes"]

                        cer_16 = geo_physical_grp.variables["cer_16"][:].astype(float)
                        cer_21 = geo_physical_grp.variables["cer_21"][:].astype(float)
                        cer_22 = geo_physical_grp.variables["cer_22"][:].astype(float)
                        cot = geo_physical_grp.variables["cot_16"][:].astype(float)
                        cth = geo_physical_grp.variables["cth"][:].astype(float)
                        cwp = geo_physical_grp.variables["cwp_16"][:].astype(float)
                        lat = geo_loc_grp.variables["latitude"][:]
                        lon = geo_loc_grp.variables["longitude"][:]
                        cloud_flag = geo_physical_grp.variables["water_cloud"][:].astype(int)  # 1=cloud, 0=no cloud
                        sza = scan_grp.variables["csol_z"][:].astype(float)
                        sza = sza[:, None] * np.ones_like(lon)

                        # Mask invalid pixels
                        mask = np.zeros(lat.shape, dtype=bool)
                        for var in [cer_16, cer_21, cer_22, cot, sza]:
                            mask |= (var < 0) | np.isnan(var)

                        lon_diff = np.abs(np.diff(lon, axis=1))
                        discontinuity = lon_diff > 180
                        discontinuity = np.hstack([np.zeros((discontinuity.shape[0], 1), dtype=bool), discontinuity])
                        mask |= discontinuity
                        mask |= (cer_16 < 0)

                        if not world_data:
                            mask |= ~(
                                (lat >= lat_min) & (lat <= lat_max) &
                                (lon >= lon_min) & (lon <= lon_max)
                            )

                        valid = ~mask

                        # Compute cloud coverage in boxes and map back to each valid pixel
                        lat_valid = lat[valid]
                        lon_valid = lon[valid]
                        cloud_valid = cloud_flag[valid]

                        cloud_count, _, _ = np.histogram2d(lat_valid, lon_valid, bins=[lat_bins, lon_bins], weights=cloud_valid)
                        pixel_count, _, _ = np.histogram2d(lat_valid, lon_valid, bins=[lat_bins, lon_bins])

                        cloud_coverage_box = np.zeros_like(cloud_count)
                        cloud_coverage_box[pixel_count > 0] = cloud_count[pixel_count > 0] / pixel_count[pixel_count > 0]

                        lat_idx = np.clip(np.digitize(lat_valid, lat_bins) - 1, 0, cloud_coverage_box.shape[0]-1)
                        lon_idx = np.clip(np.digitize(lon_valid, lon_bins) - 1, 0, cloud_coverage_box.shape[1]-1)
                        cloud_coverage_pixel = cloud_coverage_box[lat_idx, lon_idx].astype(np.float32)
                        # Save variables and cloud coverage
                        for key, arr in zip(["cer_16", "cer_21", "cer_22", "cot", "sza", "lat", "lon", "cth", "cwp", "cloud_coverage"],
                                            [cer_16, cer_21, cer_22, cot, sza, lat, lon, cth, cwp, cloud_coverage_pixel]):
                            valid_values = arr[valid].ravel().astype(np.float32) if key != "cloud_coverage" else cloud_coverage_pixel
                            prev_size = dsets[key].shape[0]
                            dsets[key].resize(prev_size + len(valid_values), axis=0)
                            dsets[key][prev_size:] = valid_values

                        # Save date
                        prev_size_date = dsets["date"].shape[0]
                        dsets["date"].resize(prev_size_date + len(valid_values), axis=0)
                        dsets["date"][prev_size_date:] = file_date

                    gc.collect()

print(f"OCI Data with per-pixel cloud coverage saved incrementally to: {h5_path}")