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
    iqr_k=5,
    num_leaves=128,
    learning_rate=0.01,
    num_boost_round=300,
    progress_period=20
):
    # ---- Remove outliers only from TRAIN ----
    Q1, Q3 = np.percentile(y_train, 25), np.percentile(y_train, 75)
    IQR = Q3 - Q1
    lower, upper = Q1 - iqr_k * IQR, Q3 + iqr_k * IQR
    mask = (y_train >= lower) & (y_train <= upper)
    X_train_clean, y_train_clean = X_train[mask], y_train[mask]

    # ---- LightGBM datasets ----
    train_data = lgb.Dataset(X_train_clean, label=y_train_clean)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # ---- LightGBM parameters ----
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "verbose": 1,  # Needed for callbacks to print properly
        "force_col_wise": True
    }

    # ---- Train with early stopping + percentage progress bar ----
    tqdm_callback = TQDMProgress(total_rounds=num_boost_round)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            tqdm_callback
        ]
    )

    best_iter = model.best_iteration

    # ---- Predictions ----
    y_train_pred = model.predict(X_train_clean, num_iteration=best_iter)
    y_test_pred = model.predict(X_test, num_iteration=best_iter)

    # ---- Metrics ----
    r2_train = r2_score(y_train_clean, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return model, y_test_pred, r2_train, r2_test, mae_test, rmse_test

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

