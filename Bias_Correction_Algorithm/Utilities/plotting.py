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
    



def plot_scatter(rad_OCI_after_regression,
                 rad_OCI_before_regression,
                 rad_OCI_before_qm,
                 best_pred,
                 sf_test,
                 save_folder,
                 name):
    x = rad_OCI_after_regression
    y = rad_OCI_before_regression
    z = rad_OCI_before_qm

    # Compute metrics
    r2_xy = np.corrcoef(x, y)[0, 1] ** 2
    r2_zy = np.corrcoef(z, y)[0, 1] ** 2
    r2_sf = np.corrcoef(best_pred, sf_test)[0, 1] ** 2

    mae_xy = np.mean(np.abs(y - x))
    mae_zy = np.mean(np.abs(y - z))
    mae_sf = np.mean(np.abs(sf_test - best_pred))

    # -----------------------------
    # Plot setup
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # -----------------------------
    # Plot 1: Regression vs After QM
    # -----------------------------
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    hb = axes[0].hexbin(x, y, gridsize=100, cmap='Blues', mincnt=1, bins='log')
    axes[0].plot([min_val, max_val], [min_val, max_val], '--', color='gray', lw=1)
    axes[0].set_title(f'Regression Quality\nR²={r2_xy:.2f}')
    axes[0].set_xlabel('CER After Regression')
    axes[0].set_ylabel('CER After QM')
    axes[0].set_xlim(min_val, max_val)
    axes[0].set_ylim(min_val, max_val)
    axes[0].set_aspect('equal', adjustable='box')
    fig.colorbar(hb, ax=axes[0], label='Density')

    # -----------------------------
    # Plot 2: Before QM vs After QM
    # -----------------------------
    min_val = min(y.min(), z.min())
    max_val = max(y.max(), z.max())
    hb = axes[1].hexbin(z, y, gridsize=100, cmap='Oranges', mincnt=1,bins = 'log')
    axes[1].plot([min_val, max_val], [min_val, max_val], '--', color='gray', lw=1)
    axes[1].set_title(f'QM Quality\nR²={r2_zy:.2f}')
    axes[1].set_xlabel('CER Before QM')
    axes[1].set_ylabel('CER After QM')
    axes[1].set_xlim(min_val, max_val)
    axes[1].set_ylim(min_val, max_val)
    axes[1].set_aspect('equal', adjustable='box')
    fig.colorbar(hb, ax=axes[1], label='Density')

    # -----------------------------
    # Plot 3: SF After Regression vs SF After QM
    # -----------------------------
    min_val = min(best_pred.min(), sf_test.min())
    max_val = max(best_pred.max(), sf_test.max())
    hb = axes[2].hexbin(best_pred, sf_test, gridsize=100, cmap='Reds', mincnt=1, bins = 'log')
    axes[2].plot([min_val, max_val], [min_val, max_val], '--', color='gray', lw=1)
    axes[2].set_title(f'SF Regression Quality\nR²={r2_sf:.2f}')
    axes[2].set_xlabel('SF After Regression')
    axes[2].set_ylabel('SF After QM')
    axes[2].set_xlim(min_val, max_val)
    axes[2].set_ylim(min_val, max_val)
    axes[2].set_aspect('equal', adjustable='box')
    fig.colorbar(hb, ax=axes[2], label='Density')

    plt.tight_layout()
    try:
        plt.savefig(rf"{save_folder}\{name}")
    except:
        print(name, "plot was not saves successfully. Check save path")
    plt.show()