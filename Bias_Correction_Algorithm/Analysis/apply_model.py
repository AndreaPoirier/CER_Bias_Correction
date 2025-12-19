import os, sys, random, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.user_input import *
from utilities.processing import *
from utilities.plotting import *
from utilities.statistics import *
from import_lib import *



##################################
########## LOADING DATA ##########
##################################

# Load all data into numpy arrays
processed_data_file = folder_path_saved_processed_data + '\processed_data.h5'

with h5py.File(processed_data_file, 'r') as f:

    # OCI/MODIS
    lat = f['lat_centers'][:]
    lon = f['lon_centers'][:]

    cloud_coverage = f['cloud_coverage'][:]
    cer_16 = f['rad_uncorrected'][:]
    cer_21 = f['cer_21'][:]
    cer_22 = f['cer_22'][:]
    cth = f['cth'][:]
    cwp = f['cwp'][:]
    sza = f['sza'][:]
    chi = f['chi'][:]
    cot = f['cot'][:]
    instrument = f['Instrument_to_correct'][()].decode('utf-8')

    # HARP2
    rad_HARP2 = f['rad_HARP2'][:]
    lon_HARP2 = f['lon_HARP2'][:]
    lat_HARP2 = f['lat_HARP2'][:]

    # Scaling Factor
    sf = f['sf'][:]

plot_map_cer = True
plot_hist_cer = True
plot_cdf_cer = True
plot_map_bias = True
plot_hist_bias = True
plot_corr_analysis = True
plot_residuals = True

# Define variables to be tried in the regression
features = [cer_16, cer_21, cer_22, sza, chi, cloud_coverage, cot, cth, cwp]
features_names = ["CER 16", "CER 21","CER 22", "SZA","Chi","Cloud Coverage","COT","CTH","CWP"]

X_all = np.column_stack(features)

########################################y
##### Apply Model to Test Region #######
########################################


os.makedirs(folder_train, exist_ok=True) 
save_path = os.path.join(folder_train, "model_year.txt")  
loaded_model = lgb.Booster(model_file=save_path)


print("===== APPLYING MODEL  ======")
description_of_script = """This script takes the processed data from processed_data.h5 and applies a pre-trained ML model to it. 
"""
print(textwrap.fill(description_of_script, width=100))
print("========================================================================")
print(f"Loading {instrument} processed data from {processed_data_file}")
print(f"Loading HARP2 (for comparison) data from {processed_data_file}")
print(f"Loading trained model from {folder_train}")
print(f"Saving plots to {folder_output}")
print("========================================================================")
print(f"Region box: "f"lat_min: {lat.min():.2f} | "f"lat_max: {lat.max():.2f} | "f"lon_min: {lon.min():.2f} | "f"lon_max: {lon.max():.2f}")
print("========================================================================")
print("Plotting the following:\n")
if plot_map_cer:
    print("- Map of CER before and after correction")
if plot_hist_cer:
    print("- Histograms of CER for uncorrected, qm corrected, regressed corrected and reference")
if plot_cdf_cer:
    print("- CDF of CER for uncorrected, qm corrected, regressed corrected and reference")
if plot_map_bias:
    print("- MAP of CER bias for uncorrected and regressed corrected")
if plot_hist_bias:
    print("- Histograms of CER bias for uncorrected, qm corrected, and regressed corrected")
if plot_scatter:
    print("- Correlation analysis")
if plot_residuals:
    print("- Residuals vs variables plot before and after correction")
starting_code()


sf_pred = predict_with_tqdm(loaded_model, X_all)

rad_OCI_before_qm = X_all[:, 0]
rad_OCI_after_regression = X_all[:, 0] * sf_pred
rad_OCI_before_regression = X_all[:, 0] * sf

################################################################
########## COMPUTE DIFFERENCE BEFORE AND AFTER CORRECTION ######
################################################################

print("\n== Computing comparison statistics ...\n")

#Aggregate data
agg_uncorr, counts_uncorr, mask_uncorr = aggregate_oci_to_harp2_radius(
    lat, lon, rad_OCI_before_qm,
    lat_HARP2, lon_HARP2, radius_km=5
)
agg_corr, counts_corr, mask_corr = aggregate_oci_to_harp2_radius(
    lat, lon, rad_OCI_after_regression,
    lat_HARP2, lon_HARP2, radius_km=5
)
agg_before_reg, counts_before_reg, mask_before_reg = aggregate_oci_to_harp2_radius(
    lat, lon, rad_OCI_before_regression,
    lat_HARP2, lon_HARP2, radius_km=5
)

agg_sza, counts_sza, mask_sza = aggregate_oci_to_harp2_radius(
    lat, lon, sza,
    lat_HARP2, lon_HARP2, radius_km=5
)

agg_chi, counts_chi, mask_chi = aggregate_oci_to_harp2_radius(
    lat, lon, chi,
    lat_HARP2, lon_HARP2, radius_km=5
)

agg_cloud_cov, counts_cloud_cov, mask_cloud_cov = aggregate_oci_to_harp2_radius(
    lat, lon, cloud_coverage,
    lat_HARP2, lon_HARP2, radius_km=5
)

agg_cwp, counts_cwp, mask_cwp = aggregate_oci_to_harp2_radius(
    lat, lon, cwp,
    lat_HARP2, lon_HARP2, radius_km=5
)

agg_cth, counts_cth, mask_cth = aggregate_oci_to_harp2_radius(
    lat, lon, cth,
    lat_HARP2, lon_HARP2, radius_km=5
)

agg_cot, counts_cot, mask_cot = aggregate_oci_to_harp2_radius(
    lat, lon, cot,
    lat_HARP2, lon_HARP2, radius_km=5
)

# Combine all masks to select valid HARP2 points with OCI coverage
valid_mask = (
    mask_uncorr & mask_corr & mask_before_reg & ~np.isnan(rad_HARP2) &
    mask_sza & mask_chi & mask_cloud_cov & mask_cwp & mask_cth & mask_cot
)

# Apply mask to all paired variables
paired_harp = rad_HARP2[valid_mask]
paired_uncorr = agg_uncorr[valid_mask]
paired_corr = agg_corr[valid_mask]
paired_before_reg = agg_before_reg[valid_mask]

paired_sza = agg_sza[valid_mask]
paired_chi = agg_chi[valid_mask]
paired_cloud_cov = agg_cloud_cov[valid_mask]
paired_cwp = agg_cwp[valid_mask]
paired_cth = agg_cth[valid_mask]
paired_cot = agg_cot[valid_mask]

# Calculate and print stats
count_uncorr, rmse_uncorr, mae_uncorr, bias_uncorr, aare_uncorr, error_uncorr = print_pair_stats(paired_harp, paired_uncorr, "Uncorrected OCI data")
count_before_reg, rmse_before_reg, mae_before_reg, bias_before_reg, aare_before_reg, error_before_reg = print_pair_stats(paired_harp, paired_before_reg, "Corrected OCI Before Regression")
count_corr, rmse_corr, mae_corr, bias_corr, aare_corr, error_after_reg = print_pair_stats(paired_harp, paired_corr, "Corrected OCI After Regression")

####################################
##### SCATTER PLOTS ################
####################################

if plot_corr_analysis:
    print("\n== Plotting the correlation analysis scatter ...")
    plot_scatter(rad_OCI_after_regression,
                rad_OCI_before_regression,
                rad_OCI_before_qm,
                sf_pred,
                sf,
                save_folder=folder_output,
                name = "plot_scatter")


######################################
##### PLOT region CDF and Histo ######
######################################


if plot_hist_cer:
    print("\n== Plotting the CER histograms ...")

    plot_multiple_histograms(
            datasets=[rad_HARP2,rad_OCI_after_regression,rad_OCI_before_qm, rad_OCI_before_regression],
            labels=['HARP2','OCI After Regression','OCI Before QM','OCI Before Regression'],
            bins=100,
            title='Histograms of CER ',
            outline_only=True,
            show_stats=False,
            save_folder=folder_output,
            name = 'plot_cer_histogram',
            xtitle = 'CER (um)'
        )
    
if plot_cdf_cer:
    print("\n== Plotting the CER CDFs ...")
    plot_multiple_cdfs_with_p(
        datasets=[rad_HARP2,rad_OCI_after_regression,rad_OCI_before_qm, rad_OCI_before_regression],
        labels=['HARP2','OCI After Regression','OCI Before QM','OCI Before Regression'],
        bins=100,
        title='Cumulative Distribution Functions of CER',
        show_stats=False,
        save_folder=folder_output,
        name = "plot_cer_cdf",
        xtitle = 'CER (um)'
        )


######################################
##### PLOT error Histo ##############
######################################
if plot_hist_bias:
    print("\n== Plotting the CER bias histograms ...")
    plot_multiple_histograms(
            datasets=[error_uncorr,error_before_reg,error_after_reg],
            labels=['OCI Before QM','OCI Before Regression','OCI After Regression'],
            bins=100,
            title='Error Histogram ',
            outline_only=True,
            show_stats=False,
            save_folder=r"Bias_Correction_Algorithm\output\regression",
            name = "plot_histogram_cer_error",
            xtitle = 'CER absolute error (um)'
        )

################################################################
########## MAP OF DIFFERENCE BEFORE AND AFTER CORRECTION #######
################################################################


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
titles = ['Bias for Uncorrected OCI', 'Bias for Corrected OCI']

if plot_map_bias:
    print("\n== Plotting bias on a map ...")
    plot_pixel_level(lon_list, 
                    lat_list, 
                    radius_list, 
                    pixel_sizes, 
                    titles, 
                    lon_HARP2.min(), 
                    lon_HARP2.max(), 
                    lat_HARP2.min(), 
                    lat_HARP2.max(),
                    save_folder=r"Bias_Correction_Algorithm\output\regression",
                    name = "map_cer_bias",
                    bar_title = 'Absolute Difference in CER (um) for test data')

##################################################################
########## MAP OF CER for CORRECTED, UNCORRECTED  ################
##################################################################

if plot_map_cer:
    print("\n== Plotting CER on a map ...")

    titles = ['CER Uncorrected OCI', 'CER Corrected OCI']
    data_list = [paired_uncorr, paired_corr]
    lon_list = [lon_plot_uncorr, lon_plot_corr]
    lat_list = [lat_plot_uncorr, lat_plot_corr]

    plot_pixel_level(lon_list, 
                    lat_list, 
                    data_list, 
                    pixel_sizes, 
                    titles, 
                    lon_HARP2.min(), 
                    lon_HARP2.max(), 
                    lat_HARP2.min(), 
                    lat_HARP2.max(),
                    save_folder = r"Bias_Correction_Algorithm\output\regression",
                    name = "map_cer",
                    bar_title = 'CER (um) for test data')

#############################################
########## RESIDUAL VS VARIBALES PLOTS ######
#############################################

if plot_residuals:
    variables = [paired_sza, paired_chi, paired_cloud_cov, paired_cwp, paired_cth, paired_cot]
    labels = ['SZA', 'CHI', 'Cloud Coverage', 'CWP', 'CTH', 'COT']
    plot_residual_vs_variable(variables, 
                              labels, 
                              paired_harp, 
                              paired_corr, 
                              paired_uncorr,
                              name = 'plot_residual',
                              folder = folder_output)
