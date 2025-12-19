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
plot_l2_it = True

# Define variables to be tried in the regression
variables = [cer_16, cer_21, cer_22, sza, chi, cloud_coverage, cot, cth, cwp]
variables_names = ["CER 16", "CER 21","CER 22", "SZA","Chi","Cloud Coverage","COT","CTH","CWP"]

print("===== TRAIN MACHINE LEARNING REGRESSOR  ======")
description_of_script = """This script takes the processed data from processed_data.h5 and uses it to train the LightGBM Model. 
"""
print(textwrap.fill(description_of_script, width=100))
print("========================================================================")
print(f"Loading OCI processed data from {processed_data_file}")
print(f"Loading HARP2 (for comparison) data from {processed_data_file}")
print(f"Saving outputs (plots, model ...) to {folder_train}")
print("========================================================================")
print(f"Size of training data: {len(lat):,}")
print(f"Features used in training {variables_names}")
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
if plot_l2_it:
    print("- LightGBM model training l2 vs iterations") 

starting_code()



#######################################
##### SPLIT TRAIN AND TEST DATA #######
#######################################



train_idx, test_idx = split_train_test_data(sf,lat,lon)

y_train = sf[train_idx]
y_test  = sf[test_idx]
sf_test = sf[test_idx]
X_all = np.column_stack(variables)
X_train_full = X_all[train_idx, :]
X_test_full  = X_all[test_idx, :]

lat_OCI_test = lat[test_idx]
lon_OCI_test = lon[test_idx]

#######################################
##### CREATE .h5 FILE #################
#######################################

os.makedirs(folder_train, exist_ok=True)
results_file = os.path.join(folder_train, "regression_results_year.h5")
with h5py.File(results_file, "w") as f:
    f.create_dataset("names", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
    f.create_dataset("r2_test", shape=(0,), maxshape=(None,), dtype="f")
    f.create_dataset("mae_test", shape=(0,), maxshape=(None,), dtype="f")
    f.create_dataset("best_pred", shape=(0,), maxshape=(None,), dtype="f")
    f.attrs["best_mae"] = np.inf
    f.attrs["best_rmse"] = 0
    f.attrs["best_combo"] = ""


#######################################
##### TRAIN USING ALL VARIABLES #######
#######################################

# Use ALL variables (single combination)
sel_vars = list(range(len(variables)))
names = variables_names

X_train = X_train_full[:, sel_vars]
X_test  = X_test_full[:, sel_vars]

try:
    model, y_pred_test, r2_train, r2_test, mae_test, rmse_test = scaling_factor_model_lgbm_train_test(
        X_train, y_train, X_test, y_test,
        plot_metrics=plot_l2_it, save_folder=folder_train, name = "plot_l2_vs_iterations"
    )
    
   
    with h5py.File(results_file, "a") as f:
        f["names"].resize((1,))
        f["r2_test"].resize((1,))
        f["mae_test"].resize((1,))

        f["names"][0] = ", ".join(names)
        f["r2_test"][0] = r2_test
        f["mae_test"][0] = mae_test

        f.attrs["best_mae"] = mae_test
        f.attrs["best_rmse"] = rmse_test
        f.attrs["best_combo"] = ", ".join(names)

        f["best_pred"].resize(y_pred_test.shape)
        f["best_pred"][:] = y_pred_test

    # Save model
    save_path = os.path.join(folder_train, "model_year.txt")
    model.save_model(save_path)

except Exception as e:
    print(f"Training failed, error: {e}")
with h5py.File(results_file, "a") as f:
    if "names" in f:
        del f["names"]
    dt = h5py.string_dtype("utf-8")
    f.create_dataset("names", data=np.array(variables_names, dtype=dt))

#######################################
##### LOAD RESULTS AND SUMMARIZE ######
#######################################

print("\nRegression is done ...")

with h5py.File(results_file, "r") as f:
    names_all = f["names"][:]
    r2_all = f["r2_test"][:]
    mae_all = f["mae_test"][:]
    best_combo = f.attrs["best_combo"]
    best_mae = f.attrs["best_mae"]
    best_pred = f["best_pred"][:]



print("\nML PERFORMANCE METRIC:")
print(f"Features: {best_combo}")
print(f"MAE of Test: {best_mae:.4f}")
print(f"R2 Test {r2_all[0]:.4f}")

#######################################
##### APPLY CORRECTION ################
#######################################

rad_OCI_before_qm = X_test_full[:, 0]
rad_OCI_after_regression = X_test_full[:, 0] * best_pred
rad_OCI_before_regression = X_test_full[:, 0] * sf_test

####################################
########## Saving train data #######
####################################

# Folder and file name
filepath = os.path.join(folder_train, "train_data.h5")

# Ensure folder exists
os.makedirs(folder_train, exist_ok=True)

# Write to .h5 file
with h5py.File(filepath, "w") as f:
    f.create_dataset("X_train", data = X_train)

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


# Calculate and print stats
count_uncorr, rmse_uncorr, mae_uncorr, bias_uncorr, aare_uncorr, error_uncorr = print_pair_stats(paired_harp, paired_uncorr, "Uncorrected OCI data")
count_before_reg, rmse_before_reg, mae_before_reg, bias_before_reg, aare_before_reg, error_before_reg = print_pair_stats(paired_harp, paired_before_reg, "Corrected OCI Before Regression")
count_corr, rmse_corr, mae_corr, bias_corr, aare_corr, error_after_reg = print_pair_stats(paired_harp, paired_corr, "Corrected OCI After Regression")

# Write results to txt file
log_metrics(
    label = ["Uncorrected OCI data","Corrected OCI Before Regression","Corrected OCI After Regression"], 
    count = [count_uncorr, count_before_reg, count_corr], 
    rmse = [rmse_uncorr, rmse_before_reg, rmse_corr], 
    mae = [mae_uncorr, mae_before_reg, mae_corr], 
    bias = [bias_uncorr,bias_before_reg, bias_corr], 
    aare = [aare_uncorr, aare_before_reg, aare_corr], 
    output_file = folder_train + "\log_results_regression.txt", 
    mode="w",
    best_combo = best_combo,
    best_mae = best_mae,
    r2_all = r2_all
)


####################################
##### SCATTER PLOTS ################
####################################

if plot_corr_analysis:
    print("\n== Plotting the correlation analysis scatter ...")
    plot_scatter(rad_OCI_after_regression,
                rad_OCI_before_regression,
                rad_OCI_before_qm,
                best_pred,
                sf_test,
                save_folder=folder_train,
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
            save_folder=folder_train,
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
        save_folder=folder_train,
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


