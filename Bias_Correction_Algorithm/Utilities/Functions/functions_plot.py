import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from import_lib import *

####################################################
################# PLOTS FUNCTIONS ##################
####################################################

def plot_multiple_cdfs_with_p(
    datasets, 
    labels=None, 
    bins=200,
    xlabel='Effective Radius (µm)',
    ylabel='Cumulative Probability',
    title='Cumulative Distribution Comparison',
    colors=None,
    linewidth=2,
    alpha=0.9,
    show_stats=True,
    save_folder = r"C:\Users\andreap\Documents\GitHub\SRON\Bias_Correction_Algorithm\Output",
    name = 'CDF',
    xtitle = 'Variable'
):
    """
    Plot normalized cumulative distribution functions (CDFs) for multiple datasets,
    showing the KS-test p-values (vs the first dataset) next to each label in the legend.

    Parameters
    ----------
    datasets : list of array-like
        List of numeric arrays (e.g., radius values from different instruments).
    labels : list of str
        Labels for each dataset. If None, numbered automatically.
    bins : int or sequence
        Number of bins or bin edges for computing CDFs.
    xlabel, ylabel, title : str
        Axis labels and plot title.
    colors : list of color strings
        Optional list of colors. Defaults to matplotlib’s color cycle.
    linewidth : float
        Line width for CDF curves.
    alpha : float
        Line transparency.
    show_stats : bool
        If True, prints mean, std, and median for each dataset.
    """

    n = len(datasets)
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(n)]
    if colors is None:
        colors = plt.cm.tab10.colors
    if len(colors) < n:
        colors = list(colors) * (n // len(colors) + 1)

    # Combine all data for consistent bin edges
    all_data = np.concatenate([np.ravel(np.asarray(d)) for d in datasets if len(d) > 0])
    all_data = all_data[np.isfinite(all_data)]
    edges = np.histogram_bin_edges(all_data, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(8, 5))

    # Compute and plot all CDFs
    cdfs = []
    for data in datasets:
        data = np.asarray(data)
        data = data[np.isfinite(data)]
        hist, _ = np.histogram(data, bins=edges, density=True)
        cdf = np.cumsum(hist)
        cdf /= cdf[-1]
        cdfs.append(cdf)

    # Compute KS p-values relative to first dataset
    p_values = [None]  # first curve = reference
    KS_values = []
    for i in range(1, n):
        ks_stat, p_val = ks_2samp(datasets[0], datasets[i])
        p_values.append(p_val)
        KS_values.append(ks_stat)
    # Plot all curves with updated labels including p-values
    for i, cdf in enumerate(cdfs):
        if i == 0:
            label = f"{labels[i]} (ref)"
        else:
            label = f"{labels[i]} (p ={p_values[i]:.3f})"
        plt.plot(centers, cdf, label=label, color=colors[i], lw=linewidth, alpha=alpha,linestyle = '--')

        if show_stats:
            print(f"{labels[i]:<15} mean={np.mean(datasets[i]):6.2f}, "
                  f"std={np.std(datasets[i]):6.2f}, median={np.median(datasets[i]):6.2f}")
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    try:
        plt.savefig(rf"{save_folder}\{name}")
    except:
        print(name, "plot was not saves successfully. Check save path")

    plt.show()
 

def plot_multiple_histograms(
    datasets, 
    labels=None, 
    bins=200,
    xlabel='Effective Radius (µm)',
    ylabel='Normalized Frequency',
    title='Histogram Comparison',
    colors=None,
    linewidth=2,
    alpha=0.9,
    show_stats=True,
    outline_only=True,
    save_folder = r"C:\Users\andreap\Documents\GitHub\SRON\Bias_Correction_Algorithm\Output",
    name = 'Histogram',
    xtitle = 'Variable'
):
    """
    Plot normalized histograms for multiple datasets, 
    with an option to show outlines or filled bars.

    Parameters
    ----------
    datasets : list of array-like
        List of numeric arrays (e.g., radius values from different instruments).
    labels : list of str
        Labels for each dataset. If None, numbered automatically.
    bins : int or sequence
        Number of bins or bin edges for computing histograms.
    outline_only : bool
        If True, plot only outlines (no filled bins).
        If False, plot filled histograms (like traditional bar charts).
    """
    n = len(datasets)
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(n)]
    
    if colors is None:
        colors = list(plt.cm.tab10.colors)
    if len(colors) < n:
        colors = colors * (n // len(colors) + 1)

    # Combine all data for consistent bins
    all_data = np.concatenate([np.ravel(np.asarray(d)) for d in datasets if len(d) > 0])
    all_data = all_data[np.isfinite(all_data)]
    edges = np.histogram_bin_edges(all_data, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(8, 5))

    for i, data in enumerate(datasets):
        data = np.asarray(data)
        data = data[np.isfinite(data)]
        if len(data) == 0:
            continue

        hist, _ = np.histogram(data, bins=edges, density=True)

        if outline_only:
            plt.plot(centers, hist, label=labels[i],
                     color=colors[i % len(colors)], lw=linewidth, alpha=alpha)
        else:
            plt.hist(data, bins=edges, density=True, histtype='stepfilled',
                     alpha=0.4, color=colors[i % len(colors)], 
                     edgecolor='black', linewidth=0.5, label=labels[i])

        # Print stats if enabled
        if show_stats:
            mean, std, median = np.mean(data), np.std(data), np.median(data)
            print(f"{labels[i]:<15} mean = {mean:6.2f}, std = {std:6.2f}, median = {median:6.2f}")

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    try:
        plt.savefig(rf"{save_folder}\{name}")
    except:
        print(name, "plot was not saves successfully. Check save path")
    plt.show()
    
    

def plot_pixel_level(lon_list, lat_list, data_list, pixel_list, titles, lon_min, lon_max, lat_min, lat_max, save_folder, name, bar_title):
    # --- Create subplot figure ---
    fig, axes = plt.subplots(
        1, len(data_list),
        figsize=(15, 5),
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True
    )

    # --- Shared color scale using 25th and 75th percentiles ---
    all_data = np.concatenate([r[~np.isnan(r)] for r in data_list])
    vmin = np.percentile(all_data, 25)
    vmax = np.percentile(all_data, 75)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    for ax, lon, lat, rad, pix, title in zip(axes, lon_list, lat_list, data_list, pixel_list, titles):
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
    fig.colorbar(sm, ax=axes, orientation='vertical', label= bar_title)
    try:
        plt.savefig(rf"{save_folder}\{name}")
    except:
        print(name, "plot was not saves successfully. Check save path")
    plt.show()
    

def plot_taylor_diagram(std_devs, correlation, labels, title="Taylor Diagram"):
    """
    Plots a normalized Taylor Diagram using standard Matplotlib.
    """
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, polar=True)

    # Reference is always at (1.0, 1.0) in normalized plot
    
    # 1. Setup Grid (Correlation lines)
    r_locs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
    theta_locs = np.arccos(r_locs)
    gl = ax.grid(False) 
    
    # Radial lines (Correlation)
    for theta, label_val in zip(theta_locs, r_locs):
        ax.plot([0, theta], [0, 1.6], 'k--', alpha=0.3)
        ax.text(theta, 1.7, str(label_val), ha='center', va='bottom', fontsize=10)
    
    # Arcs (Standard Deviation)
    for r in [0.5, 1.0, 1.5]:
        x = np.linspace(0, np.pi/2, 100)
        style = 'k-' if r == 1.0 else 'k--'
        width = 2 if r == 1.0 else 1
        ax.plot(x, [r]*100, style, linewidth=width, alpha=0.5)

    # RMSE Contours (Green) - centered on (1, 0)
    rs = np.linspace(0, 2.0, 100)
    ts = np.linspace(0, np.pi/2, 100)
    R, T = np.meshgrid(rs, ts)
    X = R * np.cos(T)
    Y = R * np.sin(T)
    RMSE = np.sqrt((X - 1)**2 + Y**2) # Distance from reference (1.0, 0)
    
    contour = ax.contour(T, R, RMSE, levels=[0.2, 0.4, 0.6, 0.8, 1.0], colors='green', alpha=0.4, linestyles=':')
    ax.clabel(contour, inline=1, fontsize=10, fmt='%.1f')

    # 2. Plot Data Points
    # Convert Correlation to Angle (Theta)
    theta_data = np.arccos(correlation)
    
    # Define markers and colors
    markers = ['*', 'o', 'o', 'o']
    colors = ['k', 'r', 'b', 'g'] # Ref=Black, Orig=Red, Corr=Blue
    sizes = [200, 100, 100, 100]

    for i in range(len(std_devs)):
        ax.plot(theta_data[i], std_devs[i], marker=markers[i], markersize=np.sqrt(sizes[i]), 
                color=colors[i], label=labels[i], linestyle='None', markeredgecolor='k')

    # Styling
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_ylim(0, 1.7)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    
    # Manual Axes
    x_axis = np.linspace(0, np.pi/2, 100)
    ax.plot(x_axis, [1.7]*100, 'k-', linewidth=1) 
    ax.plot([0, 0], [0, 1.7], 'k-', linewidth=1)
    ax.plot([np.pi/2, np.pi/2], [0, 1.7], 'k-', linewidth=1)
    
    # Labels
    ax.text(np.pi/4, 1.9, "Correlation Coefficient", ha='center', rotation=-45, fontsize=12, weight='bold')
    ax.text(0, -0.1, "Normalized Standard Deviation", ha='center', fontsize=12, weight='bold')
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.title(title, y=1.15, fontsize=14)
    return fig

# ==========================================
# PART 2: Calculation Logic
# ==========================================

def calculate_taylor_stats(ref, pred):
    """
    Calculates Correlation and Standard Deviation for a single model vs reference.
    Handles NaNs automatically.
    """
    # 1. Flatten and Mask NaNs (Crucial for satellite data)
    mask = ~np.isnan(ref) & ~np.isnan(pred)
    clean_ref = ref[mask]
    clean_pred = pred[mask]

    # 2. Calculate Stats
    # Correlation (index [0,1] of the correlation matrix)
    R = np.corrcoef(clean_ref, clean_pred)[0, 1]
    
    # Standard Deviation (ddof=1 for sample std dev)
    std_dev = np.std(clean_pred, ddof=1)
    
    return std_dev, R, np.std(clean_ref, ddof=1)
