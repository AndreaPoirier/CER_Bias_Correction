import os, sys, random, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.user_input import *
from utilities.processing import *
from utilities.plotting import *
from utilities.statistics import *
from import_lib import *

# Set to True plot to be shown
plot_beeswarm = True
plot_bar = True
plot_interaction = True
plot_waterfall = True

print("===== SHAP Analysis  ======")
description_of_script = """This script takes the results from the training of the model and performs a SHAP analysis. 
"""
print(textwrap.fill(description_of_script, width=100))
print("========================================================================")
print(f"Loading pre-trained model from {folder_train}")
print(f"Saving plots to {folder_SHAP}")
print("========================================================================")
print("Plotting the following:\n")
if plot_beeswarm:
    print("- Beeswarm plot of all features")
if plot_bar:
    print("- Bar plot of mean |SHAP values|")
if plot_interaction:
    print("- Interaction values (summary plot)")
if plot_waterfall:
    print("- Waterfall plot for a sample prediction")
print("========================================================================")

# Give option or not to run code
starting_code()

# Loading train data
train_data_file = os.path.join(folder_train, "train_data.h5")  
with h5py.File(train_data_file, "r") as f:
    X_train = f["X_train"][:]

# Loading model 
model_file = os.path.join(folder_train, "model_year.txt")  
model = lgb.Booster(model_file=model_file)

# Sampling data to 1,000,000 points
n_data = 1_000_000
if X_train.shape[0] > n_data:
    idx = np.random.choice(X_train.shape[0], n_data, replace=False)
    X_train = X_train[idx]
    print("=== Sampling the data. The shape of the train data matrix now is ", X_train.shape)
else:
    print("=== No sampling.")

# Loading features names, R2 and MAE
results_file = os.path.join(folder_train, "regression_results_year.h5")
with h5py.File(results_file, "r") as f:
    feature_list = list(f["names"].asstr()[:])
    r2_all = f["r2_test"][:]
    best_mae = f.attrs["best_mae"]

r2_max = np.max(r2_all)

print("\n=== Calculating SHAP values with the following features\n")
print(feature_list)

# Computes shap values in blocks and progress bar
shap_values = calculate_shap_with_progress(model, X_train, feature_names=feature_list, batch_size=5000)

##############################
####### BEESWARM PLOT ########
##############################
if plot_beeswarm:
    print("\n=== Plotting Beeswarm plot")
    plt.figure(figsize=(12, 6))
    plt.title(f"SHAP Analysis | MAE test = {best_mae:.3f} | R2 test = {r2_max:.3f}", fontsize=16)
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(rf"{folder_SHAP}\plot_beeswarm")
    plt.show()

##############################
######## BAR PLOT ############
##############################
if plot_bar:
    print("\n=== Plotting Bar plot")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(rf"{folder_SHAP}\plot_shap_bar")
    plt.show()



##############################
#### WATERFALL PLOT ##########
##############################
if plot_waterfall:
    print("\n=== Plotting Waterfall plot for a sample prediction")
    sample_idx = random.randint(0, X_train.shape[0] - 1)
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[sample_idx], show=False)
    plt.tight_layout()
    plt.savefig(rf"{folder_SHAP}\plot_shap_waterfall_sample")
    plt.show()



##############################
## INTERACTION VALUES ########
##############################
if plot_interaction:
    print("\n=== Calculating and plotting interaction values (top 4 features). Note this may take a long time. ")
    # Compute interaction values
    explainer = shap.TreeExplainer(model)
    shap_interaction_values = explainer.shap_interaction_values(X_train)

    # Compute mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top4_idx = np.argsort(mean_abs_shap)[-4:]  # indices of top 4 features
    top4_features = [feature_list[i] for i in top4_idx]

    # Plot summary only for top 4 features
    shap.summary_plot(
        shap_interaction_values, 
        X_train, 
        feature_names=feature_list,
        max_display=4,       # show only top 4 features
        show=False
    )
    plt.tight_layout()
    plt.savefig(rf"{folder_SHAP}\plot_shap_interaction_top4")
    plt.show()