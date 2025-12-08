import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utilities.user_input import *
from Utilities.Functions.functions_data_processing import *
from Utilities.Functions.functions_plot import *
from Utilities.Functions.functions_statistics import *
from import_lib import *
from Data_Processing.get_HARP2 import *
from Data_Processing.get_OCI import *


# ===================================================
# LOAD OCI & HARP2 DATA PROGRESSIVELY
# ===================================================
output_folder = folder_path_saved_processed_data

OCI_h5 = h5py.File(os.path.join(output_folder, "OCI_data.h5"), "r")
HARP2_h5 = h5py.File(os.path.join(output_folder, "HARP2_data.h5"), "r")

lat_OCI = OCI_h5["lat"]
lon_OCI = OCI_h5["lon"]
rad_OCI = OCI_h5["radius"]
cot_OCI = OCI_h5["cot"]
sza_OCI = OCI_h5["sza"]

lat_HARP2 = HARP2_h5["lat"]
lon_HARP2 = HARP2_h5["lon"]
rad_HARP2 = HARP2_h5["radius"]

print(f"OCI total points: {len(lat_OCI):,}")
print(f"HARP2 total points: {len(lat_HARP2):,}\n")

# Box half-size in degrees
box_half_deg = 0.5

# Prepare output lists
oci_in_box_list = []
harp_in_box_list = []
rad_OCI_to_be_corrected = []   # store the center radius for each box
rad_OCI_corrected = []   # store the center radius for each box
n_data = len(rad_OCI)
n_run = 0
lon_oci_plot = []
lat_oci_plot = []

harp2_to_plot = []
lat_harp2_to_plot = []
lon_harp2_to_plot = []


for lat_c, lon_c, r_c in zip(lat_OCI, lon_OCI, rad_OCI):
    

    lat_min_1, lat_max_1 = lat_c - box_half_deg, lat_c + box_half_deg
    lon_min_1, lon_max_1 = lon_c - box_half_deg, lon_c + box_half_deg

    
    oci_mask = (
        (lat_OCI >= lat_min_1) & (lat_OCI <= lat_max_1) &
        (lon_OCI >= lon_min_1) & (lon_OCI <= lon_max_1)
    )
   
    # HARP2 points inside the box
    harp_mask = (
        (lat_HARP2 >= lat_min_1) & (lat_HARP2 <= lat_max_1) &
        (lon_HARP2 >= lon_min_1) & (lon_HARP2 <= lon_max_1)
    )

    
    if len(rad_HARP2[harp_mask]) < 30 or len(rad_OCI[oci_mask]) < 30:
        continue
  
    fitted_rad_OCI = fit_lognormal(rad_OCI[oci_mask])
    fitted_rad_HARP2 = fit_lognormal(rad_HARP2[harp_mask])

    rad_OCI_to_be_corrected.append(r_c)  # store the radius of the central OCI point
    rad_OCI_corrected.append(quantile_mapping(r_c,fitted_rad_OCI,fitted_rad_HARP2 ))  # store the radius of the central OCI point

    lat_oci_plot.append(lat_c)
    lon_oci_plot.append(lon_c)


    
    n_run += 1
    if n_run % 3000 == 0:
        # OCI_1_deg_corrected = quantile_mapping(x = None , bias = rad_OCI[oci_mask], unbias=rad_HARP2[harp_mask])
        # plot_multiple_cdfs_with_p_new(
        #     datasets=[rad_HARP2[harp_mask],OCI_1_deg_corrected],
        #     labels=['HARP2','OCI (1 km) Corrected'],
        #     bins=100,
        #     title='Corrected CDF (1 deg bin -- '+ str(n_days) +' day(s))'
        # )
        print("Corrected " + str(round(n_run/n_data * 100,2)) + " % of data")

print("Corrected 100 % of data")
print("\n" + str(len(rad_OCI_corrected)) + " size data that can be corrected")



# Convert data to array

# OCI
lat_oci_plot = np.array(lat_oci_plot)
lon_oci_plot = np.array(lon_oci_plot)
rad_OCI_corrected = np.array(rad_OCI_corrected)
rad_OCI_to_be_corrected = np.array(rad_OCI_to_be_corrected)


# HARP2
lat_HARP2 = np.array(lat_HARP2)
lon_HARP2 = np.array(lon_HARP2)
rad_HARP2 = np.array(rad_HARP2)



######################################
##### PLOT region CDF and Histo ######
######################################

# Define map extent
plot_multiple_histograms(
        datasets=[rad_HARP2,rad_OCI_corrected,rad_OCI_to_be_corrected,],
        labels=['HARP2','OCI Corrected','OCI Uncorrected'],
        bins=100,
        title='Quantile Mapping Corrected Histogram ',
        outline_only=False,
    )

plot_multiple_cdfs_with_p(
    datasets=[rad_HARP2,rad_OCI_corrected,rad_OCI_to_be_corrected,],
    labels=['HARP2','OCI Corrected','OCI Uncorrected'],
    bins=100,
    title='Quantile Mapping Corrected CDF'
    )


####################################################
##### PLOT MAP ON THE PIXEL LEVLE FOR 3 CASES ######
####################################################

lon_list = [lon_oci_plot, lon_oci_plot, lon_HARP2]
lat_list = [lat_oci_plot, lat_oci_plot, lat_HARP2]
radius_list = [rad_OCI_to_be_corrected,rad_OCI_corrected, rad_HARP2]
pixel_sizes = [360/(2*np.pi*6371*np.cos(np.deg2rad(np.max(lat_oci_plot)))), 360/(2*np.pi*6371*np.cos(np.deg2rad(np.max(lat_oci_plot)))), 5 * 360/(2*np.pi*6371*np.cos(np.deg2rad(np.max(lat_HARP2))))]  # each dataset has its own pixel size

# --- Create subplot figure ---
fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 5),
    subplot_kw={'projection': ccrs.PlateCarree()},
    constrained_layout=True
)

# Shared color scale across all plots
norm = Normalize(
    vmin=0.25 * min(np.nanmin(r) for r in radius_list),
    vmax=0.75 * max(np.nanmax(r) for r in radius_list)
)
cmap = plt.cm.viridis
titles = ['Uncorrected OCI','Corrected OCI',  'HARP2']

for ax, lon, lat, rad, pix, title in zip(axes, lon_list, lat_list, radius_list, pixel_sizes, titles):
    # Ensure arrays
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    rad = np.asarray(rad)

    # Mask invalid data
    valid = ~np.isnan(lon) & ~np.isnan(lat) & ~np.isnan(rad)
    lon, lat, rad = lon[valid], lat[valid], rad[valid]

    
    lon_edges = np.arange(lon_min, lon_max + pix, pix)
    lat_edges = np.arange(lat_min, lat_max + pix, pix)

    grid, _, _, _ = binned_statistic_2d(
        lat, lon, rad, statistic='mean', bins=[lat_edges, lon_edges]
    )

    # Mask bins with no data
    grid = np.ma.masked_invalid(grid)

    # --- Plot with pcolormesh ---
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    mesh = ax.pcolormesh(
        lon_edges, lat_edges, grid,
        cmap=cmap, norm=norm,
        shading='auto',
        transform=ccrs.PlateCarree()
    )

    ax.set_title(f"{title} (pixel = {pix:.3f}°)")

# Shared colorbar
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=axes, orientation='vertical', label='Radius')

plt.show()




 
###############################
##### COMPUTE DIFFERENCE ######
###############################

radius_km = 5  # adjust to approximate HARP2 pixel size

agg_uncorr, counts_uncorr, mask_uncorr = aggregate_oci_to_harp2_radius(
    lat_oci_plot, lon_oci_plot, rad_OCI_to_be_corrected,
    lat_HARP2, lon_HARP2, radius_km=radius_km
)
agg_corr, counts_corr, mask_corr = aggregate_oci_to_harp2_radius(
    lat_oci_plot, lon_oci_plot, rad_OCI_corrected,
    lat_HARP2, lon_HARP2, radius_km=radius_km
)

# Valid HARP2 points with OCI coverage
valid_mask = mask_uncorr & mask_corr & ~np.isnan(rad_HARP2)
paired_harp = rad_HARP2[valid_mask]
paired_uncorr = agg_uncorr[valid_mask]
paired_corr = agg_corr[valid_mask]

# -------------------------
# Print statistics
# -------------------------
print_pair_stats(paired_harp, paired_uncorr, "Uncorrected (aggregated)")
print_pair_stats(paired_harp, paired_corr, "Corrected (aggregated)")
