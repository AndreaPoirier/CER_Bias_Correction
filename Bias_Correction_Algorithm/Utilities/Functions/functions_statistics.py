import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from import_lib import *


def compute_Lorg_OII(lat, lon, cloud_mask, r_values, max_points=1000):
    """
    Compute dimensionless (normalized) L_org and OII from cloud mask.

    Parameters
    ----------
    lat, lon : 1D arrays
        Latitude and longitude (degrees) of pixel centers.
    cloud_mask : 1D array (0 or 1)
        1 where cloudy, 0 otherwise.
    r_values : array
        Radii in km at which to evaluate L̂(r). The largest value defines r_max.
    max_points : int, optional
        Maximum number of cloudy points used (for memory control).

    Returns
    -------
    r_values : array
        Radii in km.
    Lhat : array
        Empirical L̂(r) (km).
    L_org_dimless : float
        Dimensionless L_org (integral of deviation, signed).
    OII_dimless : float
        Dimensionless OII (RMS of deviation, always ≥ 0).
    """

    # --- 1. Extract cloudy pixels ---
    idx = np.where(cloud_mask == 1)[0]
    if len(idx) < 2:
        raise ValueError("Need at least 2 cloudy points.")
    lat_cloud, lon_cloud = lat[idx], lon[idx]
    Nc = len(lat_cloud)

    # --- 2. Optional subsampling ---
    if Nc > max_points:
        sel = np.random.choice(Nc, max_points, replace=False)
        lat_cloud = lat_cloud[sel]
        lon_cloud = lon_cloud[sel]
        Nc = len(lat_cloud)

    # --- 3. Convert lat/lon to Cartesian (km) ---
    R = 6371.0
    lat0, lon0 = np.mean(lat_cloud), np.mean(lon_cloud)
    x = R * np.cos(np.radians(lat0)) * np.radians(lon_cloud - lon0)
    y = R * np.radians(lat_cloud - lat0)
    coords = np.vstack((x, y)).T

    # --- 4. KD-tree for efficient neighbor search ---
    tree = cKDTree(coords)
    Lx, Ly = np.ptp(x), np.ptp(y)
    A = Lx * Ly

    # --- 5. Compute empirical L̂(r) ---
    Lhat = np.zeros_like(r_values, dtype=float)
    for k, r in enumerate(r_values):
        pairs = tree.query_pairs(r)
        count_pairs = len(pairs)
        Lhat[k] = np.sqrt((A / (np.pi * Nc * (Nc - 1))) * (2 * count_pairs))

    # --- 6. Normalize to dimensionless quantities ---
    r_max = r_values[-1]
    r_tilde = r_values / r_max
    Lhat_tilde = Lhat / r_max

    # --- 7. Compute dimensionless L_org and OII ---
    diff_tilde = Lhat_tilde - r_tilde
    L_org_dimless = np.trapz(diff_tilde, r_values) * 1/r_max

    OII_dimless = np.sqrt(np.trapz(diff_tilde**2, r_values) * 1/r_max)

    return r_values, Lhat, L_org_dimless, OII_dimless

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
    learning_rate=0.05,
    num_boost_round=250,
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

    return model, y_test_pred, r2_train, r2_test, mae_test

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




class ScalingFactorNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_nn(X_train, y_train, X_test, y_test, epochs=20, batch_size=1024, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Incremental StandardScaler for huge datasets
    scaler = StandardScaler()
    # Fit scaler in batches to avoid memory issues
    for start in range(0, X_train.shape[0], batch_size * 10):  # 10x batch for faster scaling
        end = min(start + batch_size * 10, X_train.shape[0])
        scaler.partial_fit(X_train[start:end])
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    
    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model, optimizer, loss
    model = ScalingFactorNN(X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        # tqdm for progress
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for X_batch, y_batch in loop:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
            loop.set_postfix(loss=loss.item())
        avg_loss = epoch_loss / X_train.shape[0]
        train_losses.append(avg_loss)
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_t.to(device)).cpu().numpy().flatten()
    
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    
    return model, y_pred_test, r2_test, mae_test

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


def synchronized_random_sample(
    list_of_arrays, 
    sample_fraction, 
    replace = False
) :
    """
    Performs synchronized random sampling across a list of 1D arrays or lists 
    based on a desired fraction of the total data size.
    
    This ensures that all resulting sampled arrays contain elements that correspond 
    to the same index locations in the original inputs.

    Args:
        list_of_arrays: A list of 1D arrays (numpy.ndarray) or Python lists 
                        that must all have the same length.
        sample_fraction: The fraction of the data to include in the sample (float between 0.0 and 1.0).
        replace: If True, samples are drawn with replacement. If False (default), 
                 samples are drawn without replacement.

    Returns:
        A new list of numpy.ndarray objects, where each array is the sampled version 
        of the corresponding input array.
    """

    # Check that all arrays have the same length
    first_length = len(list_of_arrays[0])

    # 2. Calculate the sample size from the fraction
    sample_size = int(first_length * sample_fraction)

    # 3. Generate the universal set of sampled indices
    original_indices = np.arange(first_length)
    sampled_indices = np.random.choice(
        a=original_indices,
        size=sample_size,
        replace=replace
    )
    
    # 4. Apply the same sampled indices to all input arrays
    sampled_arrays = [arr[sampled_indices] for arr in list_of_arrays]
    
    return sampled_arrays

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

