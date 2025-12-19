import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from import_lib import *


# Custom tqdm callback for LightGBM
class TQDMProgress:
    def __init__(self, total_rounds):
        self.pbar = tqdm(total=total_rounds, unit="iter", ncols=100, desc="Training Progress")

    def __call__(self, env: CallbackEnv):
        self.pbar.update(1)
        if env.iteration + 1 == env.end_iteration:
            self.pbar.close()
def scaling_factor_model_lgbm_train_test(
    X_train, y_train, X_test, y_test, 
    num_leaves=128,
    learning_rate=0.01,
    num_boost_round=250,
    progress_period=20,
    plot_metrics=True,
    save_folder="save_folder",
    name="name",
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import lightgbm as lgb
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    # ---- LightGBM datasets ----
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # ---- Parameters (multiple built-in metrics) ----
    params = {
        "objective": "regression",
        "metric": ["l2", "rmse", "l1"],  # ✅ Added RMSE & MAE
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "verbose": -1,
        "force_col_wise": True
    }

    evals_result = {}

    # ---- Train model ----
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.record_evaluation(evals_result),
            TQDMProgress(total_rounds=num_boost_round),
        ]
    )

    best_iter = model.best_iteration

    # ---- Plot metrics ----
    if plot_metrics:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        metric_map = {
            "l2": "L2 (MSE)",
            "rmse": "RMSE",
            "l1": "MAE"
        }

        for ax, (metric, title) in zip(axs, metric_map.items()):
            ax.plot(
                evals_result["train"][metric][:best_iter],
                label="Train"
            )
            ax.plot(
                evals_result["valid"][metric][:best_iter],
                label="Validation"
            )
            ax.axvline(best_iter, linestyle="--", color="gray")
            ax.set_title(f"{title} vs Iterations")
            ax.set_xlabel("Boosting Iterations")
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        os.makedirs(save_folder, exist_ok=True)
        try:
            plt.savefig(os.path.join(save_folder, f"{name}.png"))
        except Exception as e:
            print(name, "metric plot not saved:", e)
        plt.show()


    # ---- Predictions ----
    y_train_pred = model.predict(X_train, num_iteration=best_iter)
    y_test_pred = model.predict(X_test, num_iteration=best_iter)

    # ---- Final metrics ----
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return model, y_test_pred, r2_train, r2_test, mae_test, rmse_test
def scaling_factor_model_lgbm_train_test(
    X_train, y_train, X_test, y_test, 
    num_leaves=128,
    learning_rate=0.01,
    num_boost_round=250,
    progress_period=20,
    plot_metrics=True,
    save_folder="save_folder",
    name="name",
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import lightgbm as lgb
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    os.makedirs(save_folder, exist_ok=True)

    # ---- LightGBM datasets ----
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # ---- Parameters ----
    params = {
        "objective": "regression",
        "metric": ["l2", "rmse", "l1"],  # L2, RMSE, MAE
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "verbose": -1,
        "force_col_wise": True
    }

    evals_result = {}

    # ---- Train model ----
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.record_evaluation(evals_result),
            TQDMProgress(total_rounds=num_boost_round),
        ]
    )

    best_iter = model.best_iteration

    # ================= METRICS + TREE DIAGNOSTICS =================
    if plot_metrics:

        # ---- Metrics vs iteration ----
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        metric_map = {
            "l2": "L2 (MSE)",
            "rmse": "RMSE",
            "l1": "MAE"
        }

        for ax, (metric, title) in zip(axs, metric_map.items()):
            ax.plot(evals_result["train"][metric][:best_iter], label="Train")
            ax.plot(evals_result["valid"][metric][:best_iter], label="Validation")
            ax.axvline(best_iter, linestyle="--", color="gray")
            ax.set_title(f"{title} vs Iterations")
            ax.set_xlabel("Boosting Iterations")
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"plot_metrics_vs_iteration.png"))
        plt.show()

        # ---- Tree diagnostics ----
        dump = model.dump_model()
        trees = dump["tree_info"]

        tree_depths = []
        tree_leaves = []

        def traverse(node, depth=0):
            if "split_index" not in node:
                return depth, 1
            ld, ll = traverse(node["left_child"], depth + 1)
            rd, rl = traverse(node["right_child"], depth + 1)
            return max(ld, rd), ll + rl

        for tree in trees:
            depth, leaves = traverse(tree["tree_structure"])
            tree_depths.append(depth)
            tree_leaves.append(leaves)

        iters = np.arange(1, len(tree_depths) + 1)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        axs[0].plot(iters, tree_depths)
        axs[0].set_title("Tree Depth vs Boosting Iteration")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Tree Depth")
        axs[0].grid(True)

        axs[1].plot(iters, tree_leaves)
        axs[1].set_title("Leaves per Tree vs Boosting Iteration")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Number of Leaves")
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"plot_tree_diagnostics.png"))
        plt.show()

    # ================= END DIAGNOSTICS =================

    # ---- Predictions ----
    y_train_pred = model.predict(X_train, num_iteration=best_iter)
    y_test_pred = model.predict(X_test, num_iteration=best_iter)

    # ---- Final metrics ----
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return model, y_test_pred, r2_train, r2_test, mae_test, rmse_test

def split_train_test_data(sf_list,lat_centers,lon_centers):
    """
    Split data into spatially independent training and test sets
    using latitude–longitude block sampling.

    Parameters
    ----------
    sf_list : array-like
        Target or data array used to determine sample count
    lat_centers : array-like
        Latitude values of samples
    lon_centers : array-like
        Longitude values of samples

    Returns
    -------
    train_idx : numpy.ndarray
        Indices of training samples
    test_idx : numpy.ndarray
        Indices of test samples
    """

    all_idx = np.arange(len(sf_list))

    # ---- Spatial binning parameters ----
    # Choose bin size according to spatial correlation length
    # Example: 1 degree bins (adjust if needed)
    lat_bin_size = 1.0
    lon_bin_size = 1.0

    lat_bins = np.floor(lat_centers / lat_bin_size).astype(int)
    lon_bins = np.floor(lon_centers / lon_bin_size).astype(int)

    # Unique spatial block ID
    spatial_block_id = lat_bins * 100000 + lon_bins

    # ---- Select spatial blocks for test set ----
    rng = np.random.default_rng(42)
    unique_blocks = np.unique(spatial_block_id)

    n_test_blocks = int(0.2 * len(unique_blocks))
    test_blocks = rng.choice(unique_blocks, size=n_test_blocks, replace=False)

    test_mask = np.isin(spatial_block_id, test_blocks)
    train_mask = ~test_mask

    train_idx = all_idx[train_mask]
    test_idx  = all_idx[test_mask]
    return train_idx, test_idx


def quantile_mapping(x=None, bias=None, unbias=None):
    """
    Apply quantile mapping to correct a biased distribution.

    If x is given, return the corrected x value(s).
    If x is None, return the corrected version of the full biased distribution.

    Parameters
    ----------
    x : float, array-like, or None
        Value(s) to transform. If None, correct the full biased distribution.
    bias : array-like
        Biased/source distribution.
    unbias : array-like
        Reference/unbiased distribution.

    Returns
    -------
    transformed : float or np.ndarray
        If x is given: corrected value(s) matching the distribution of 'unbias'.
        If x is None: full corrected biased distribution.
    """

    # Sort to create empirical CDFs
    sorted_bias = np.sort(bias)
    sorted_unbias = np.sort(unbias)

    # Empirical cumulative probabilities
    p_bias = np.linspace(0, 1, len(sorted_bias))
    p_unbias = np.linspace(0, 1, len(sorted_unbias))

    # If no x given → correct full bias distribution
    if x is None:
        quantiles = np.interp(sorted_bias, sorted_bias, p_bias)
        corrected_bias = np.interp(quantiles, p_unbias, sorted_unbias)
        return corrected_bias

    # Else, correct given x
    x_arr = np.asarray(x, dtype=np.float64)
    original_shape = x_arr.shape
    x_arr = x_arr.ravel()

    quantiles = np.interp(x_arr, sorted_bias, p_bias)
    transformed = np.interp(quantiles, p_unbias, sorted_unbias)

    transformed = transformed.reshape(original_shape)

    if np.isscalar(x):
        return float(transformed)
    return transformed


def compute_chi(cot_array):
    """
    Compute chi = exp(mean(log(x))) / mean(x) safely.
    Handles NaNs, zeros, and negative/invalid values gracefully.
    """
    # Remove NaN and non-positive values (log undefined for ≤ 0)
    cot_valid = cot_array[np.isfinite(cot_array) & (cot_array > 0)]
    if len(cot_valid) == 0:
        return np.nan

    # Compute geometric and arithmetic means safely
    log_mean = np.mean(np.log(cot_valid))
    arithmetic_mean = np.mean(cot_valid)

    # Avoid division by zero
    if arithmetic_mean <= np.finfo(np.float32).eps:
        return np.nan

    chi = np.exp(log_mean) / arithmetic_mean
    return np.float32(chi)



def fit_lognormal(data):
    """
    Fit a 1D array to a lognormal distribution and return the fitted array
    with the same size as the input.
    
    Parameters:
        data (array-like): Input 1D data (must be positive values)
        
    Returns:
        fitted_data (np.ndarray): Fitted lognormal values (same length as input)
        params (tuple): (shape, loc, scale) parameters of the fitted distribution
    """
    data = np.asarray(data)
    
    # Remove non-positive values (lognormal is defined for x > 0)
    data = data[data > 0]
    
    # Fit to lognormal distribution
    shape, loc, scale = lognorm.fit(data, floc=0)  # loc fixed to 0 for simplicity
    
    # Generate fitted values (using quantiles that match the sorted data)
    sorted_data = np.sort(data)
    cdf_vals = np.linspace(0, 1, len(sorted_data), endpoint=False)[1:]  # avoid 0 and 1
    fitted_vals = lognorm.ppf(cdf_vals, shape, loc=loc, scale=scale)
    
    # Interpolate to match original data order
    fitted_data = np.interp(np.argsort(np.argsort(data)), np.arange(len(fitted_vals)), fitted_vals)
    
    return fitted_data

def calculate_shap_with_progress(model, data, feature_names=None, batch_size=10000):
    """
    Calculates SHAP values in batches to handle large datasets.
    FIX: Accepts feature_names argument because numpy arrays don't have .columns
    """
    explainer = shap.TreeExplainer(model)
    shap_batches = []
    
    # Create batches
    num_batches = int(np.ceil(len(data) / batch_size))
    batches = np.array_split(data, num_batches)
    
    print(f"Processing {len(data)} rows in {num_batches} batches...")
    
    for batch in tqdm(batches, desc="SHAP values calculation progress"):
        # Calculate SHAP for this batch
        batch_values = explainer(batch)
        shap_batches.append(batch_values)
        
    # Concatenate results
    combined_values = np.concatenate([b.values for b in shap_batches], axis=0)
    combined_base = np.concatenate([b.base_values for b in shap_batches], axis=0)
    combined_data = np.concatenate([b.data for b in shap_batches], axis=0)
    
    # FIX: Use the passed list of names instead of data.columns
    return shap.Explanation(combined_values, base_values=combined_base, data=combined_data, feature_names=feature_names)

def predict_with_tqdm(model, X, batch_size=100_000):
    preds = []
    for i in tqdm(range(0, len(X), batch_size), desc="Predicting"):
        batch = X[i:i + batch_size]
        preds.append(model.predict(batch))
    return np.concatenate(preds)