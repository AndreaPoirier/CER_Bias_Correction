import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utilities.user_input import *
from Utilities.Functions.functions_data_processing import *
from Utilities.Functions.functions_statistics import *
from import_lib import *



################################
#### LOADING PROCESSED DATA ####
################################

# Input files
output_folder = folder_path_saved_processed_data
OCI_path = os.path.join(output_folder, "OCI_data.h5")
HARP2_path = os.path.join(output_folder, "HARP2_data.h5")
OCI_h5 = h5py.File(OCI_path, "r")
HARP2_h5 = h5py.File(HARP2_path, "r")

# Loading OCI data
lat_OCI_d = OCI_h5["lat"]
lon_OCI_d = OCI_h5["lon"]
rad_OCI_d = OCI_h5["cer_16"]
cot_OCI_d = OCI_h5["cot"]
sza_OCI_d = OCI_h5["sza"]
cloud_coverage_d = OCI_h5["cloud_coverage"]
date_OCI_d = OCI_h5["date"]
rad_OCI_21_d = OCI_h5["cer_21"]
rad_OCI_22_d = OCI_h5["cer_22"]
cth_OCI_d = OCI_h5["cth"]
cwp_OCI_d = OCI_h5["cwp"]

# Loading HARP2 data
lat_H_d = HARP2_h5["lat"]
lon_H_d = HARP2_h5["lon"]
rad_H_d = HARP2_h5["radius"]
date_H_d = HARP2_h5["date"]



##################################
#### CREATING OUTPUT .h5 FILE ####
##################################

output_file = os.path.join(folder_path_saved_processed_data, "processed_data.h5")
if os.path.exists(output_file):
    os.remove(output_file)

with h5py.File(output_file, "w") as f:
    ds_names = ["lat_centers", "lon_centers", "chi", "sf", "rad_uncorrected",
                "sza", "cot", "lat_harp2", "lon_harp2", "rad_harp2", "cloud_coverage",
                "cer_21", "cer_22", "cth", "cwp"]
    for name in ds_names:
        f.create_dataset(name, shape=(0,), maxshape=(None,), dtype='float32',
                         chunks=(50000,), compression='gzip', compression_opts=4)
        
    f.create_dataset("lat_HARP2", data=lat_H_d[:].astype(np.float32), dtype='float32', compression='gzip')
    f.create_dataset("lon_HARP2", data=lon_H_d[:].astype(np.float32), dtype='float32', compression='gzip')
    f.create_dataset("rad_HARP2", data=rad_H_d[:].astype(np.float32), dtype='float32', compression='gzip')
    
    dt = h5py.string_dtype(encoding='utf-8')
    f.create_dataset('Instrument_to_correct', data='OCI', dtype=dt)



######################################
#### PRINT RUN CONFIGURATION #########
######################################

print("\n===== PROCESSING DATA (get_processed_data.py) ======")
description_of_script = """This script pre-processes the data by applying the Quantile Mapping method to the OCI data using HARP2 data. 
It does so by dividing the data into days, such that only data from the same day is processed together. To avoid, memory issues the computation
is performed chunk wise. Note that the run time may be long.
"""
print(textwrap.fill(description_of_script, width=100))
print("========================================================================")
print(f"Loading OCI data from {output_folder}\OCI_data.h5")
print(f"Loading HARP2 data from {output_folder}\HARP2_data.h5")
print("========================================================================")
print(f"OCI total points: {len(lat_OCI_d):,}")
print(f"HARP2 total points: {len(lat_H_d):,}")
print("========================================================================")
print(f"Saving OCI processed data to {output_file}")
print(f"Saving HARP2 processed data to {output_file}")
print(f"Processed data that will be outputted {ds_names}")
print("========================================================================")
print(f"OCI and HARP2 block size: {block_size:,}")
print(f"OCI chunk size: {OCI_chunk_size:,}")
print(f"HARP2 chunk size: {HARP2_chunk_size:,}")
print(f"Box width: {half * 2} degree")
print("========================================================================")

# Giving option to user to run or not the code
starting_code()



########################################################
#### FUNCTION FOR CODE AND MEMORY OPTIMIZATION #########
########################################################

# Numba optimizer helpers
@nb.njit(fastmath=True)
def compute_bbox_fast(lat, lon, half):
    lat_min = lat.min() - half
    lat_max = lat.max() + half
    lon_min = lon.min() - half
    lon_max = lon.max() + half
    return lat_min, lat_max, lon_min, lon_max


# Buffer for output data
class FastBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.size = 0
        self.data = {
            'lat': np.empty(capacity, dtype=np.float32),
            'lon': np.empty(capacity, dtype=np.float32),
            'chi': np.empty(capacity, dtype=np.float32),
            'sf': np.empty(capacity, dtype=np.float32),
            'rad': np.empty(capacity, dtype=np.float32),
            'sza': np.empty(capacity, dtype=np.float32),
            'cot': np.empty(capacity, dtype=np.float32),
            'latH': np.empty(capacity, dtype=np.float32),
            'lonH': np.empty(capacity, dtype=np.float32),
            'radH': np.empty(capacity, dtype=np.float32),
            'cloud_coverage': np.empty(capacity, dtype=np.float32),
            'cer_21': np.empty(capacity, dtype=np.float32),
            'cer_22': np.empty(capacity, dtype=np.float32),
            'cth': np.empty(capacity, dtype=np.float32),
            'cwp': np.empty(capacity, dtype=np.float32)
        }

    def append(self, **kwargs):
        if self.size >= self.capacity:
            self._expand()
        idx = self.size
        for key, val in kwargs.items():
            self.data[key][idx] = val
        self.size += 1

    def _expand(self):
        new_capacity = self.capacity * 2
        for key in self.data:
            new_arr = np.empty(new_capacity, dtype=np.float32)
            new_arr[:self.capacity] = self.data[key]
            self.data[key] = new_arr
        self.capacity = new_capacity

    def clear(self):
        self.size = 0

    def get_slice(self):
        return {key: arr[:self.size] for key, arr in self.data.items()}

buffer = FastBuffer(capacity=100000)
total_saved = 0

# Flush function
def flush():
    global total_saved
    if buffer.size == 0:
        return

    data_slice = buffer.get_slice()

    with h5py.File(output_file, "a") as f:
        size = f["lat_centers"].shape[0]
        newsize = size + buffer.size

        mapping = {
            'lat_centers': 'lat', 'lon_centers': 'lon', 'chi': 'chi', 'sf': 'sf',
            'rad_uncorrected': 'rad', 'sza': 'sza', 'cot': 'cot',
            'lat_harp2': 'latH', 'lon_harp2': 'lonH', 'rad_harp2': 'radH',
            'cloud_coverage': 'cloud_coverage', 'cer_21': 'cer_21', 'cer_22': 'cer_22',
            'cth': 'cth', 'cwp': 'cwp'
        }

        for h5_key, buf_key in mapping.items():
            f[h5_key].resize((newsize,))
            f[h5_key][size:newsize] = data_slice[buf_key]

    total_saved += buffer.size
    buffer.clear()
    gc.collect()


# HARP2 collection function
def collect_harp_for_bbox_and_day(lat_min, lat_max, lon_min, lon_max, day,
                                  harp_lat_d=lat_H_d, harp_lon_d=lon_H_d, harp_date_d=date_H_d,
                                  harp_rad_d=rad_H_d, chunk=600_000):
    result_lists = {'lat': [], 'lon': [], 'rad': []}
    N = len(harp_date_d)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        dates = harp_date_d[start:end]
        normalized = normalize_date_batch(dates)
        date_mask = (normalized == day)
        if not date_mask.any():
            continue
        global_idx = np.where(date_mask)[0] + start
        lat_block = harp_lat_d[global_idx].astype(np.float32)
        lon_block = harp_lon_d[global_idx].astype(np.float32)
        bbox_mask = ((lat_block >= lat_min) & (lat_block <= lat_max) & 
                     (lon_block >= lon_min) & (lon_block <= lon_max))
        if bbox_mask.any():
            final_idx = global_idx[bbox_mask]
            result_lists['lat'].append(lat_block[bbox_mask])
            result_lists['lon'].append(lon_block[bbox_mask])
            result_lists['rad'].append(harp_rad_d[final_idx].astype(np.float32))
    if not result_lists['lat']:
        return (np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), 
                np.empty(0, dtype=np.float32))
    return (np.concatenate(result_lists['lat']),
            np.concatenate(result_lists['lon']),
            np.concatenate(result_lists['rad']))


###########################
#### FINDING DAYS #########
###########################

print("\n==> Scanning input files for unique days (streaming)...")
days_OCI = get_unique_days_stream(date_OCI_d, chunk=OCI_chunk_size)
days_H = get_unique_days_stream(date_H_d, chunk=HARP2_chunk_size)
all_days = sorted(set(days_OCI) & set(days_H))
print(f"Found {len(all_days)} common day(s) to process\n")


###############################
#### PROCESSING LOOPS #########
###############################

print("==> Starting processing loop...\n")
for day in tqdm(all_days, desc="Days"):
    print(f"\n--- Day: {day} ---")
    N_oci = len(date_OCI_d)

    for oci_chunk_start in tqdm(range(0, N_oci, OCI_chunk_size), desc=f"Chunks for day {day}", leave=False):
        oci_chunk_end = min(oci_chunk_start + OCI_chunk_size, N_oci)
        load_start = max(0, oci_chunk_start - overlap_rows)
        load_end = min(N_oci, oci_chunk_end + overlap_rows)

        # Load date and filter for day
        date_block = date_OCI_d[load_start:load_end]
        normalized_dates = normalize_date_batch(date_block)
        local_match_positions = [i for i, d in enumerate(normalized_dates) if d == day]
        if not local_match_positions:
            continue

        # Load full arrays for matched indices
        lat_block_all = lat_OCI_d[load_start:load_end].astype(np.float32)
        lon_block_all = lon_OCI_d[load_start:load_end].astype(np.float32)
        rad_block_all = rad_OCI_d[load_start:load_end].astype(np.float32)
        cot_block_all = cot_OCI_d[load_start:load_end].astype(np.float32)
        sza_block_all = sza_OCI_d[load_start:load_end].astype(np.float32)
        cloud_block_all = cloud_coverage_d[load_start:load_end].astype(np.float32)
        cer_21_block_all = rad_OCI_21_d[load_start:load_end].astype(np.float32)
        cer_22_block_all = rad_OCI_22_d[load_start:load_end].astype(np.float32)
        cth_block_all = cth_OCI_d[load_start:load_end].astype(np.float32)
        cwp_block_all = cwp_OCI_d[load_start:load_end].astype(np.float32)

        global_indices = np.array(load_start + np.array(local_match_positions, dtype=np.int64), dtype=np.int64)
        lat_matched = lat_block_all[local_match_positions]
        lon_matched = lon_block_all[local_match_positions]
        rad_matched = rad_block_all[local_match_positions]
        cot_matched = cot_block_all[local_match_positions]
        sza_matched = sza_block_all[local_match_positions]
        cloud_matched = cloud_block_all[local_match_positions]
        cer_21_matched = cer_21_block_all[local_match_positions]
        cer_22_matched = cer_22_block_all[local_match_positions]
        cth_matched = cth_block_all[local_match_positions]
        cwp_matched = cwp_block_all[local_match_positions]

        num_points = len(global_indices)
        num_blocks = int(np.ceil(num_points / block_size))

        # Process per block
        for bi in range(num_blocks):
            bstart = bi * block_size
            bend = min((bi + 1) * block_size, num_points)

            lat_blk = lat_matched[bstart:bend]
            lon_blk = lon_matched[bstart:bend]
            rad_blk = rad_matched[bstart:bend]
            cot_blk = cot_matched[bstart:bend]
            sza_blk = sza_matched[bstart:bend]
            cloud_blk = cloud_matched[bstart:bend]
            cer_21_blk = cer_21_matched[bstart:bend]
            cer_22_blk = cer_22_matched[bstart:bend]
            cth_blk = cth_matched[bstart:bend]
            cwp_blk = cwp_matched[bstart:bend]

            # Compute bbox
            lat_min, lat_max, lon_min, lon_max = compute_bbox_fast(lat_blk, lon_blk, half)

            # Collect HARP2 neighbors
            harp_lat, harp_lon, harp_rad = collect_harp_for_bbox_and_day(
                lat_min, lat_max, lon_min, lon_max, day,
                harp_lat_d=lat_H_d, harp_lon_d=lon_H_d, 
                harp_date_d=date_H_d, harp_rad_d=rad_H_d,
                chunk=HARP2_chunk_size
            )

            if harp_lat.size == 0:
                continue

            # Build KDTrees
            coords_harp = np.column_stack((harp_lat, harp_lon))
            tree_harp = cKDTree(coords_harp, leafsize=leafsize_for_kdtree)
            coords_oci = np.column_stack((lat_matched, lon_matched))
            tree_oci = cKDTree(coords_oci, leafsize=leafsize_for_kdtree)

            coords_query = np.column_stack((lat_blk, lon_blk))
            harp_neighbors_list = tree_harp.query_ball_point(coords_query, diag, workers=-1)
            oci_neighbors_list = tree_oci.query_ball_point(coords_query, diag, workers=-1)

            valid_mask = np.array([len(h) >= min_neighbors and len(o) >= min_neighbors 
                                   for h, o in zip(harp_neighbors_list, oci_neighbors_list)])
            if not valid_mask.any():
                del tree_harp, tree_oci
                gc.collect()
                continue

            valid_indices = np.where(valid_mask)[0]

            for i_local in valid_indices:
                latc = lat_blk[i_local]
                lonc = lon_blk[i_local]
                r0 = rad_blk[i_local]
                cloud_val = cloud_blk[i_local]
                cot_val = cot_blk[i_local]
                cer_21_val = cer_21_blk[i_local]
                cer_22_val = cer_22_blk[i_local]
                cth_val = cth_blk[i_local]
                cwp_val = cwp_blk[i_local]

                # HARP neighbors
                idx_harp = harp_neighbors_list[i_local]
                harp_subset_lat = harp_lat[idx_harp]
                harp_subset_lon = harp_lon[idx_harp]
                harp_subset_rad = harp_rad[idx_harp]

                mask_H = ((harp_subset_lat >= latc - half) & (harp_subset_lat <= latc + half) &
                          (harp_subset_lon >= lonc - half) & (harp_subset_lon <= lonc + half))
                if mask_H.sum() < min_neighbors:
                    continue

                rad_H_masked = harp_subset_rad[mask_H]
                lat_H_masked = harp_subset_lat[mask_H]
                lon_H_masked = harp_subset_lon[mask_H]

                # OCI neighbors
                idx_oci = oci_neighbors_list[i_local]
                oci_subset_lat = lat_matched[idx_oci]
                oci_subset_lon = lon_matched[idx_oci]
                oci_subset_rad = rad_matched[idx_oci]
                oci_subset_cot = cot_matched[idx_oci]

                mask_O = ((oci_subset_lat >= latc - half) & (oci_subset_lat <= latc + half) &
                          (oci_subset_lon >= lonc - half) & (oci_subset_lon <= lonc + half))
                if mask_O.sum() < min_neighbors:
                    continue

                rad_O_masked = oci_subset_rad[mask_O]
                cot_O_masked = oci_subset_cot[mask_O]

                try:
                    rad_O_masked = fit_lognormal(rad_O_masked)
                    rad_H_masked = fit_lognormal(rad_H_masked)
                except:
                    continue

                sf_val = quantile_mapping(r0, rad_O_masked, rad_H_masked) / r0
                chi_val = compute_chi(cot_O_masked)

                if np.isnan(chi_val) or np.isnan(sf_val):
                    continue

                buffer.append(
                    lat=latc, lon=lonc, chi=chi_val, sf=sf_val, rad=r0,
                    sza=sza_blk[i_local], cot=cot_val,
                    latH=lat_H_masked.mean(), lonH=lon_H_masked.mean(),
                    radH=rad_H_masked.mean(),
                    cloud_coverage=cloud_val,
                    cer_21=cer_21_val, cer_22=cer_22_val,
                    cth=cth_val, cwp=cwp_val
                )

            if ((bi + 1) % save_every_n_blocks) == 0 or (bi == num_blocks - 1):
                flush()

            del tree_harp, tree_oci
            gc.collect()

        del (lat_block_all, lon_block_all, rad_block_all, cot_block_all,
             sza_block_all, cloud_block_all, cer_21_block_all, cer_22_block_all,
             cth_block_all, cwp_block_all)
        del (lat_matched, lon_matched, rad_matched, cot_matched, sza_matched, 
             cloud_matched, cer_21_matched, cer_22_matched, cth_matched, cwp_matched)
        gc.collect()

    flush()
    print(f"Finished day {day}: total saved so far {total_saved:,}")

# Final flush
flush()
print(f"\n==== ALL DAYS PROCESSED — final saved entries: {total_saved:,} ====")

OCI_h5.close()
HARP2_h5.close()

