import os, sys, random, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.user_input import *
from utilities.processing import *
from utilities.plotting import *
from utilities.statistics import *
from import_lib import *


# Input files
MODIS_h5 = h5py.File(os.path.join(folder_path_saved_processed_data, "MODIS_data.h5"), "r")
HARP2_h5 = h5py.File(os.path.join(folder_path_saved_processed_data, "HARP2_data.h5"), "r")

# Loading MODIS data
lat_MODIS = MODIS_h5["lat"][:]
lon_MODIS = MODIS_h5["lon"][:]
rad_MODIS = MODIS_h5["radius"][:]
cot_MODIS = MODIS_h5["cot"][:]              
sza_MODIS = MODIS_h5["sza"][:]
date_MODIS = MODIS_h5["date"][:]
cloud_MODIS = MODIS_h5["cloud_coverage"][:]   

# Loading HARP2 data
lat_HARP2_dset = HARP2_h5["lat"]
lon_HARP2_dset = HARP2_h5["lon"]
rad_HARP2_dset = HARP2_h5["radius"]
date_HARP2 = HARP2_h5["date"][:]


# Creating output .h5 file 
os.makedirs(folder_path_saved_processed_data, exist_ok=True)
output_file = os.path.join(folder_path_saved_processed_data, "processed_data.h5")

if os.path.exists(output_file):
    os.remove(output_file)

with h5py.File(output_file, 'w') as f:
    for name in [
        "lat_centers", "lon_centers", "chi", "sf", 
        "rad_uncorrected", "sza",
        "lat_harp2", "lon_harp2", "rad_harp2",
        "cloud_coverage",      
        "cot"                 
    ]:
        f.create_dataset(name, shape=(0,), maxshape=(None,), dtype='float32',
                         chunks=(10000,), compression='gzip')

    f.create_dataset("lat_HARP2", data=lat_HARP2_dset[:], dtype='float32', compression='gzip')
    f.create_dataset("lon_HARP2", data=lon_HARP2_dset[:], dtype='float32', compression='gzip')
    f.create_dataset("rad_HARP2", data=rad_HARP2_dset[:], dtype='float32', compression='gzip')
    dt = h5py.string_dtype(encoding='utf-8')
    f.create_dataset('Instrument_to_correct', data='MODIS', dtype=dt)


#Printing parameters
print("===== PROCESSING DATA ======")
print("========================================================================")
print(f"Loading MODIS data from {folder_path_saved_processed_data}\MODIS_data.h5")
print(f"Loading HARP2 data from {folder_path_saved_processed_data}\HARP2_data.h5")
print("========================================================================")
print(f"MODIS total points: {len(lat_MODIS):,}")
print(f"HARP2 total points: {len(lat_HARP2_dset):,}")
print("========================================================================")
print("MODIS dates:", {int(d) for d in set(date_MODIS)})
print("HARP2 dates:", {int(d) for d in set(date_HARP2)})
print("========================================================================")
print(f"Saving MODIS processed data to {output_file}")
print(f"Saving HARP2 processed data to {output_file}")
print("========================================================================")
print(f"Box width: {half * 2} degrees")
print(f"Block size {MODIS_chunk_size:,} points")
print(f"Saving every {save_every_n_blocks} blocks")
print("========================================================================")

# Giving option to user to run or not the code
starting_code()


# Initialize buffers
lat_buffer, lon_buffer, chi_buffer, sf_buffer, rad_buffer, sza_buffer = [], [], [], [], [], []
lat_H_buffer, lon_H_buffer, rad_H_buffer, cloud_buffer, cot_buffer = [], [], [], [], []     



# Buffer save function
def save_buffer_to_disk():
    global total_saved
    if len(lat_buffer) == 0:
        return

    with h5py.File(output_file, 'a') as f:

        size = f["lat_centers"].shape[0]
        newsize = size + len(lat_buffer)

        for name, arr in zip(
            [
                "lat_centers","lon_centers","chi","sf","rad_uncorrected","sza",
                "lat_harp2","lon_harp2","rad_harp2",
                "cloud_coverage",
                "cot"                     # <<< NEW
            ],
            [
                lat_buffer, lon_buffer, chi_buffer, sf_buffer, rad_buffer, sza_buffer,
                lat_H_buffer, lon_H_buffer, rad_H_buffer,
                cloud_buffer,
                cot_buffer                # <<< NEW
            ]
        ):
            f[name].resize((newsize,))
            f[name][size:newsize] = np.asarray(arr, dtype=np.float32)

    total_saved += len(lat_buffer)

    # clear buffers
    lat_buffer.clear(); lon_buffer.clear(); chi_buffer.clear(); sf_buffer.clear()
    rad_buffer.clear(); sza_buffer.clear()
    lat_H_buffer.clear(); lon_H_buffer.clear(); rad_H_buffer.clear()
    cloud_buffer.clear()
    cot_buffer.clear()         # <<< NEW
    gc.collect()

# Building global KD Trees
print("Building global KD Trees...")

coords_MODIS = np.column_stack((lat_MODIS, lon_MODIS))
coords_HARP = np.column_stack((lat_HARP2_dset[:], lon_HARP2_dset[:]))

tree_MODIS = cKDTree(coords_MODIS, compact_nodes=True, balanced_tree=True)
tree_HARP = cKDTree(coords_HARP, compact_nodes=True, balanced_tree=True)

del coords_MODIS, coords_HARP
gc.collect()

# ===================================================
# GLOBAL PROCESSING
# ===================================================
N = len(lat_MODIS)
num_blocks = int(np.ceil(N / MODIS_chunk_size))
print(f"\nProcessing globally in {num_blocks} blocks...\n")
total_saved = 0
for b in tqdm(range(num_blocks), desc="Global blocks"):
    start = b * MODIS_chunk_size
    end = min((b + 1) * MODIS_chunk_size, N)

    lat_block = lat_MODIS[start:end]
    lon_block = lon_MODIS[start:end]
    rad_block = rad_MODIS[start:end]
    cot_block = cot_MODIS[start:end]     # <<< NEW
    sza_block = sza_MODIS[start:end]
    cloud_block = cloud_MODIS[start:end]

    coords_block = np.column_stack((lat_block, lon_block))

    harp_neighbors_list = tree_HARP.query_ball_point(coords_block, diag, workers=-1)
    MODIS_neighbors_list = tree_MODIS.query_ball_point(coords_block, diag, workers=-1)

    MODIS_indices = sorted({idx for group in MODIS_neighbors_list for idx in group})
    if len(MODIS_indices) == 0:
        continue

    lat_O = lat_MODIS[MODIS_indices]
    lon_O = lon_MODIS[MODIS_indices]
    rad_O = rad_MODIS[MODIS_indices]
    cot_O = cot_MODIS[MODIS_indices]           
    cloud_O = cloud_MODIS[MODIS_indices]

    idx_map = {idx: i for i, idx in enumerate(MODIS_indices)}

    # ===================================================
    # LOOP
    # ===================================================
    for i_local in range(end - start):
        r_c = rad_block[i_local]

        harp_idx = harp_neighbors_list[i_local]
        if len(harp_idx) < min_neighbors:
            continue

        lat_H_all = lat_HARP2_dset[harp_idx]
        lon_H_all = lon_HARP2_dset[harp_idx]
        rad_H_all = rad_HARP2_dset[harp_idx]

        mask_H = (
            (lat_H_all >= lat_block[i_local]-half) & (lat_H_all <= lat_block[i_local]+half) &
            (lon_H_all >= lon_block[i_local]-half) & (lon_H_all <= lon_block[i_local]+half)
        )

        rad_H_sel = rad_H_all[mask_H]
        if len(rad_H_sel) < min_neighbors:
            continue

        MODIS_idx = MODIS_neighbors_list[i_local]

        if len(MODIS_idx) < min_neighbors:
            continue

        local_idx = [idx_map[k] for k in MODIS_idx]

        rad_O_all = rad_O[local_idx]
        cot_O_all = cot_O[local_idx]      
        cloud_O_all = cloud_O[local_idx]

        mask_O = (
            (lat_O[local_idx] >= lat_block[i_local]-half) & (lat_O[local_idx] <= lat_block[i_local]+half) &
            (lon_O[local_idx] >= lon_block[i_local]-half) & (lon_O[local_idx] <= lon_block[i_local]+half)
        )

        rad_O_sel = rad_O_all[mask_O]
        cot_O_sel = cot_O_all[mask_O]    
        cloud_sel = cloud_O_all[mask_O]

        if len(rad_O_sel) < min_neighbors:
            continue

        rad_O_sel = fit_lognormal(rad_O_sel)
        rad_H_sel = fit_lognormal(rad_H_sel)

        chi = compute_chi(cot_O_sel)
        sf = quantile_mapping(r_c, rad_O_sel, rad_H_sel) / r_c

        if np.isnan(chi) or np.isnan(sf):
            continue

        # append buffers
        lat_buffer.append(lat_block[i_local])
        lon_buffer.append(lon_block[i_local])
        chi_buffer.append(chi)
        sf_buffer.append(sf)
        rad_buffer.append(r_c)
        sza_buffer.append(sza_block[i_local])

        cloud_buffer.append(np.mean(cloud_sel))
        cot_buffer.append(np.mean(cot_O_sel))   

        lat_H_buffer.append(lat_H_all[mask_H].mean())
        lon_H_buffer.append(lon_H_all[mask_H].mean())
        rad_H_buffer.append(rad_H_sel.mean())

    if (b + 1) % save_every_n_blocks == 0 or (b + 1) == num_blocks:
        save_buffer_to_disk()
        print(f" Saved {total_saved:,} results so far...")

print(f"\nGlobal processing complete — {total_saved:,} results saved to: {output_file}")
