import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utilities.user_input import *
from Utilities.Functions.functions_data_processing import *
from Utilities.Functions.functions_plot import *
from Utilities.Functions.functions_statistics import *
from import_lib import *



print("Loading train data and model\n")

# Loading train data
train_data_file = os.path.join(folder_output, "train_data.h5")
with h5py.File(train_data_file, "r") as f:
    X_train = f["X_train"][:]


# Loading model 
model_file = os.path.join(folder_output, "model.txt")  
model = lgb.Booster(model_file=model_file)

# Sampling data to 1,000,000 points
idx = np.random.choice(X_train.shape[0], 100000, replace=False)
X_train = X_train[idx]
print("The shape of the train data matrix is ", X_train.shape)

# Loading features names
results_file = os.path.join(folder_output, "regression_results.h5")
with h5py.File(results_file, "r") as f:
    feature_list = list(f["names"].asstr()[:])
    r2_all = f["r2_test"][:]
    best_mae = f.attrs["best_mae"]

r2_max = np.max(r2_all)


print("\nCalculating SHAP values with the following features\n")
print(feature_list)

# Call the function with the new feature_names argument
shap_values = calculate_shap_with_progress(model, X_train, feature_names=feature_list, batch_size=5000)

##############################
#### BEESWARM PLOT ###########
##############################

print("\nPlotting Beeswarm plot")

plt.figure(figsize=(12, 6))
plt.title(f"SHAP Analysis | MAE test = "f"{best_mae:.3f}"" | R2 test = "f"{r2_max:.3f}", fontsize=16)

shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig(r"C:\Users\andreap\Documents\Cloud_Effective_Radius_Bias_Correction_Algorithm\Bias_Correction_Algorithm\Output\Validation\SHAP1.png")
plt.show()



