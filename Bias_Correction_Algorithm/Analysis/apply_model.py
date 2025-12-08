import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utilities.user_input import *
from Utilities.Functions.functions_data_processing import *
from Utilities.Functions.functions_plot import *
from Utilities.Functions.functions_statistics import *
from import_lib import *

##################################
########## LOADING DATA ##########
##################################

# Load all data into numpy arrays
processed_data_file = folder_path_saved_processed_data + '\processed_data.h5'

with h5py.File(processed_data_file, 'r') as f:
    lat_centers = f['lat_centers'][:]
    lon_centers = f['lon_centers'][:]
    cloud_coverage = f['cloud_coverage'][:]
    cot = f['cot'][:]
    sf_list = f['sf'][:]
    rad_uncorrected_list = f['rad_uncorrected'][:]
    sza_list = f['sza'][:]
    rad_HARP2 = f['rad_HARP2'][:]
    lon_HARP2 = f['lon_HARP2'][:]
    lat_HARP2 = f['lat_HARP2'][:]
    chi_list = f['chi'][:]

# Define variables to be tried in the regression
variables = [rad_uncorrected_list, sza_list, chi_list, cloud_coverage, cot]
variables_names = ["Uncorrected Radius", "SZA","Chi","Cloud Coverage","COT"]

variables = [np.array(v) for v in variables]
sf = np.array(sf_list)
X_all = np.column_stack(variables)

########################################
##### Apply Model to Test Region #######
########################################


folder_path = r"C:\Users\andreap\Documents\Cloud_Effective_Radius_Bias_Correction_Algorithm\Bias_Correction_Algorithm\Output\Regression_with_Cloud_Coverage" 
os.makedirs(folder_path, exist_ok=True)  
save_path = os.path.join(folder_path, "model.txt")  
loaded_model = lgb.Booster(model_file=save_path)

sf_pred = loaded_model.predict(X_all)


rad_OCI_before_qm = X_all[:, 0]
rad_OCI_after_regression = X_all[:, 0] * sf_pred
rad_OCI_before_regression = X_all[:, 0] * sf
lat_OCI_test = lat_centers
lon_OCI_test = lon_centers


######################################
##### PLOT region CDF and Histo ######
######################################

print("\n== Plotting the histogram and CDF ...")

plot_multiple_histograms(
        datasets=[rad_HARP2,rad_OCI_after_regression,rad_OCI_before_qm, rad_OCI_before_regression],
        labels=['HARP2','MODIS After Regression','MODIS Before QM','MODIS Before Regression'],
        bins=100,
        title='Quantile Mapping Corrected Histogram ',
        outline_only=True,
        show_stats=False,
        save_folder=r"C:\Users\andreap\Documents\GitHub\SRON\Bias_Correction_Algorithm\Output\Regression"
    )

plot_multiple_cdfs_with_p(
    datasets=[rad_HARP2,rad_OCI_after_regression,rad_OCI_before_qm, rad_OCI_before_regression],
    labels=['HARP2','MODIS After Regression','MODIS Before QM','MODIS Before Regression'],
    bins=100,
    title='Quantile Mapping Corrected CDF',
    show_stats=False,
    save_folder=r"C:\Users\andreap\Documents\GitHub\SRON\Bias_Correction_Algorithm\Output\Regression"
    )



################################################################
########## COMPUTE DIFFERENCE BEFORE AND AFTER CORRECTION ######
################################################################

print("\n== Computing comparison statistics ...\n")

#Aggregate data
agg_uncorr, counts_uncorr, mask_uncorr = aggregate_oci_to_harp2_radius(
    lat_OCI_test, lon_OCI_test, rad_OCI_before_qm,
    lat_HARP2, lon_HARP2, radius_km=5
)
agg_corr, counts_corr, mask_corr = aggregate_oci_to_harp2_radius(
    lat_OCI_test, lon_OCI_test, rad_OCI_after_regression,
    lat_HARP2, lon_HARP2, radius_km=5
)
agg_before_reg, counts_before_reg, mask_before_reg = aggregate_oci_to_harp2_radius(
    lat_OCI_test, lon_OCI_test, rad_OCI_before_regression,
    lat_HARP2, lon_HARP2, radius_km=5
)

# Valid HARP2 points with OCI coverage
valid_mask = mask_uncorr & mask_corr & mask_before_reg & ~np.isnan(rad_HARP2)
paired_harp = rad_HARP2[valid_mask]
paired_uncorr = agg_uncorr[valid_mask]
paired_corr = agg_corr[valid_mask]
paired_before_reg = agg_before_reg[valid_mask]



count_uncorr, rmse_uncorr, mae_uncorr, bias_uncorr, aare_uncorr, error_uncorr = print_pair_stats(paired_harp, paired_uncorr, "Uncorrected OCI data")
count_before_reg, rmse_before_reg, mae_before_reg, bias_before_reg, aare_before_reg, error_before_reg = print_pair_stats(paired_harp, paired_before_reg, "Corrected OCI Before Regression")
count_corr, rmse_corr, mae_corr, bias_corr, aare_corr, error_after_reg = print_pair_stats(paired_harp, paired_corr, "Corrected OCI After Regression")


################################################################
########## SHOW CORRELATION WITH FEATURES BEFORE AND AFTER #######
################################################################

# Cloud inhomogeneity
agg_chi, counts_chi, mask_chi= aggregate_oci_to_harp2_radius(
    lat_OCI_test, lon_OCI_test, chi_list,
    lat_HARP2, lon_HARP2, radius_km=5
)
agg_chi = agg_chi[valid_mask]
r_uncorr_chi, _ = pearsonr(agg_chi, error_uncorr)
r_corr_chi, _   = pearsonr(agg_chi, error_after_reg)

# Solar Zenith Angle
agg_sza, counts_sza, mask_sza= aggregate_oci_to_harp2_radius(
    lat_OCI_test, lon_OCI_test, sza_list,
    lat_HARP2, lon_HARP2, radius_km=5
)
agg_sza = agg_sza[valid_mask]
r_uncorr_sza, _ = pearsonr(agg_sza, error_uncorr)
r_corr_sza, _   = pearsonr(agg_sza, error_after_reg)

r_uncorr_cer, _ = pearsonr(paired_uncorr, error_uncorr)
r_corr_cer, _   = pearsonr(paired_uncorr, error_after_reg)
# plots

fig, ax = plt.subplots(1, 3, figsize=(12, 5))

# CHI
ax[0].scatter(agg_chi, error_uncorr,
              label=f'Uncorrected (r = {r_uncorr_chi:.3f})')
ax[0].scatter(agg_chi, error_after_reg,
              label=f'Corrected (r = {r_corr_chi:.3f})')
ax[0].legend(loc='upper left')
ax[0].set_xlabel("Cloud Inhomogeneity (-)")
ax[0].set_ylabel("Error (um)")
ax[0].set_title("Error vs Chi")

# SZA
ax[1].scatter(agg_sza, error_uncorr,
              label=f'Uncorrected (r = {r_uncorr_sza:.3f})')
ax[1].scatter(agg_sza, error_after_reg,
              label=f'Corrected (r = {r_corr_sza:.3f})')
ax[1].legend(loc='upper left')
ax[1].set_xlabel("Solar Zenith Angle (deg)")
ax[1].set_ylabel("Error (um)")
ax[1].set_title("Error vs SZA")

# raduncorr
ax[2].scatter(paired_uncorr, error_uncorr,
              label=f'Uncorrected (r = {r_uncorr_cer:.3f})')
ax[2].scatter(paired_uncorr, error_after_reg,
              label=f'Corrected (r = {r_corr_cer:.3f})')
ax[2].legend(loc='upper left')
ax[2].set_xlabel("Uncorrected CER (um)")
ax[2].set_ylabel("Error (um)")
ax[2].set_title("Error vs uncorrected CER")

# Layout
plt.tight_layout()
plt.show()



################################################################
########## MAP OF DIFFERENCE BEFORE AND AFTER CORRECTION #######
################################################################

print("\n== Plotting bias on a map ...")

diff_uncorr = np.abs(paired_uncorr - paired_harp)
diff_corr   = np.abs(paired_corr - paired_harp)


lon_plot_uncorr = lon_HARP2[valid_mask]
lat_plot_uncorr = lat_HARP2[valid_mask]
lon_plot_corr = lon_HARP2[valid_mask]
lat_plot_corr = lat_HARP2[valid_mask]

lon_list = [lon_plot_uncorr, lon_plot_corr]
lat_list = [lat_plot_uncorr, lat_plot_corr]
radius_list = [diff_uncorr, diff_corr]

pixel_sizes = [5 * 360/(2*np.pi*6371*np.cos(np.deg2rad(np.mean(lat_HARP2)))), 5 * 360/(2*np.pi*6371*np.cos(np.deg2rad(np.mean(lat_HARP2))))]  
titles = ['Bias for Uncorrected MODIS', 'Bias for Corrected MODIS']

plot_pixel_level(lon_list, 
                 lat_list, 
                 radius_list, 
                 pixel_sizes, 
                 titles, 
                 lon_HARP2.min(), 
                 lon_HARP2.max(), 
                 lat_HARP2.min(), 
                 lat_HARP2.max(),
                 save_folder=r"C:\Users\andreap\Documents\GitHub\SRON\Bias_Correction_Algorithm\Output\Regression",
                 name = "Bias_uncorrected_and_corrected",
                 bar_title = 'Absolute Difference in CER (um) for test data')
