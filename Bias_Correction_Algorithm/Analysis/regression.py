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


#######################################
##### SPLIT TRAIN AND TEST DATA #######
#######################################

all_idx = np.arange(len(sf_list))
train_idx, test_idx, y_train, y_test = train_test_split(all_idx, sf, test_size=0.2, random_state=42)
sf_test = sf[test_idx]
X_all = np.column_stack(variables)
X_train_full = X_all[train_idx, :]
X_test_full  = X_all[test_idx, :]

lat_OCI_test = lat_centers[test_idx]
lon_OCI_test = lon_centers[test_idx]

#######################################
##### CREATE .h5 FILE #################
#######################################

os.makedirs(folder_output, exist_ok=True)
results_file = os.path.join(folder_output, "regression_results.h5")
with h5py.File(results_file, "w") as f:
    f.create_dataset("names", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
    f.create_dataset("r2_test", shape=(0,), maxshape=(None,), dtype="f")
    f.create_dataset("mae_test", shape=(0,), maxshape=(None,), dtype="f")
    f.create_dataset("best_pred", shape=(0,), maxshape=(None,), dtype="f")
    f.attrs["best_mae"] = np.inf
    f.attrs["best_combo"] = ""


#######################################
##### LOOP THROUGH COMBINATIONS #######
#######################################

total_combinations = sum(
    len(list(itertools.combinations(range(len(variables)), r))) 
    for r in range(1, len(variables)+1)
)
count = 0

for r in range(1, len(variables)+1):
    for combo_idx in itertools.combinations(range(len(variables)), r):
        sel_vars = list(combo_idx)
        names = [variables_names[i] for i in sel_vars]
        
        X_train = X_train_full[:, sel_vars]
        X_test  = X_test_full[:, sel_vars]
        count += 1

        print(f"Progress: {100*count/total_combinations:.1f}% ({count}/{total_combinations})", end="\r")

        if not len(names) == len(variables_names):
            continue

        try:
            print(names)
            model, y_pred_test, r2_train, r2_test, mae_test = scaling_factor_model_lgbm_train_test(
                X_train, y_train, X_test, y_test, iqr_k=5
            )

            with h5py.File(results_file, "a") as f:
                n = len(f["r2_test"])
                f["names"].resize((n+1,))
                f["r2_test"].resize((n+1,))
                f["mae_test"].resize((n+1,))

                f["names"][n] = ", ".join(names)
                f["r2_test"][n] = r2_test
                f["mae_test"][n] = mae_test

                # Update best prediction if better MAE found
                if mae_test < f.attrs["best_mae"]:
                    f.attrs["best_mae"] = mae_test
                    f.attrs["best_combo"] = ", ".join(names)
                    f["best_pred"].resize(y_pred_test.shape)
                    f["best_pred"][:] = y_pred_test

                    # Saving model to be used later 
                    os.makedirs(folder_output, exist_ok=True)  
                    save_path = os.path.join(folder_output, "model.txt")   
                    model.save_model(save_path)


        except Exception as e:
            print(f"\nSkipping {names}: {e}")

with h5py.File(results_file, "a") as f:
    if "names" in f:
        del f["names"]
    dt = h5py.string_dtype("utf-8")
    f.create_dataset("names", data=np.array(variables_names, dtype=dt))

#######################################
##### LOAD RESULTS AND SUMMARIZE ######
#######################################

print("\nRegression is done, starting to load best features combinations ...")

with h5py.File(results_file, "r") as f:
    names_all = f["names"][:]
    r2_all = f["r2_test"][:]
    mae_all = f["mae_test"][:]
    best_combo = f.attrs["best_combo"]
    best_mae = f.attrs["best_mae"]
    best_pred = f["best_pred"][:]



# Sort by MAE (ascending)
sorted_idx = np.argsort(mae_all)
results_sorted = [
    (names_all[i], r2_all[i], mae_all[i])
    for i in sorted_idx[:min(10, len(sorted_idx))]
]

print("\nTOP 10 VARIABLE COMBINATIONS:")
for i, (names, r2, mae) in enumerate(results_sorted, 1):
    print(f"{i}. Combo: {names} → Test MAE SF = {mae:.4f} and Test R² = {r2:.4f}")

print("\nBEST VARIABLE COMBINATION FOUND:")
print(f"Combo: {best_combo}")
print(f"Best SF Test MAE: {best_mae:.4f}")


#######################################
##### APPLY CORRECTION ################
#######################################

rad_OCI_before_qm = X_test_full[:, 0]
rad_OCI_after_regression = X_test_full[:, 0] * best_pred
rad_OCI_before_regression = X_test_full[:, 0] * sf_test





#####################################
########## Saving train data #######
#####################################

# Folder and file name
filepath = os.path.join(folder_output, "train_data.h5")

# Ensure folder exists
os.makedirs(folder_output, exist_ok=True)

# Write to .h5 file
with h5py.File(filepath, "w") as f:
    f.create_dataset("X_train", data = X_train)


######################################
##### PLOT region CDF and Histo ######
######################################

print("\n== Plotting the histogram and CDF ...")

plot_multiple_histograms(
        datasets=[rad_HARP2,rad_OCI_after_regression,rad_OCI_before_qm, rad_OCI_before_regression],
        labels=['HARP2','OCI After Regression','OCI Before QM','OCI Before Regression'],
        bins=100,
        title='Quantile Mapping Corrected Histogram ',
        outline_only=True,
        show_stats=False,
        save_folder=folder_output,
        name = 'Histograms_CER',
        xtitle = 'CER (um)'
    )

plot_multiple_cdfs_with_p(
    datasets=[rad_HARP2,rad_OCI_after_regression,rad_OCI_before_qm, rad_OCI_before_regression],
    labels=['HARP2','OCI After Regression','OCI Before QM','OCI Before Regression'],
    bins=100,
    title='Quantile Mapping Corrected CDF',
    show_stats=False,
    save_folder=folder_output,
    name = "CDF_CER",
    xtitle = 'CER (um)'
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
    output_file = folder_output + "\log_results_regression.txt", 
    mode="w",
    results_sorted = results_sorted
)

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

######################################
##### PLOT error Histo ##############
######################################

print("\n== Plotting the histogram and CDF ...")

plot_multiple_histograms(
        datasets=[error_uncorr,error_before_reg,error_after_reg],
        labels=['OCI Before QM','OCI Before Regression','OCI After Regression'],
        bins=100,
        title='Error Histogram ',
        outline_only=True,
        show_stats=False,
        save_folder=r"C:\Users\andreap\Documents\Cloud_Effective_Radius_Bias_Correction_Algorithm\Bias_Correction_Algorithm\Output\Regression",
        name = "Histograms_of_error",
        xtitle = 'CER absolute error (um)'
    )

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
titles = ['Bias for Uncorrected OCI', 'Bias for Corrected OCI']

plot_pixel_level(lon_list, 
                 lat_list, 
                 radius_list, 
                 pixel_sizes, 
                 titles, 
                 lon_HARP2.min(), 
                 lon_HARP2.max(), 
                 lat_HARP2.min(), 
                 lat_HARP2.max(),
                 save_folder=r"C:\Users\andreap\Documents\Cloud_Effective_Radius_Bias_Correction_Algorithm\Bias_Correction_Algorithm\Output\Regression",
                 name = "Bias_uncorrected_and_corrected",
                 bar_title = 'Absolute Difference in CER (um) for test data')

##################################################################
########## MAP OF CER for CORRECTED, UNCORRECTED  ################
##################################################################

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
                 save_folder = r"C:\Users\andreap\Documents\Cloud_Effective_Radius_Bias_Correction_Algorithm\Bias_Correction_Algorithm\Output\Regression",
                 name = "CER_uncorrected_and_corrected",
                 bar_title = 'CER (um) for test data')


