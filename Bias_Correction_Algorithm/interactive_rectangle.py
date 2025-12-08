import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from import_lib import *

# Global to hold the selected coordinates
selected_bbox = None

def onselect(eclick, erelease):
    """Callback when a rectangle is drawn."""
    global selected_bbox
    # Get lon/lat from click and release points
    lon1, lat1 = eclick.xdata, eclick.ydata
    lon2, lat2 = erelease.xdata, erelease.ydata

    if None in [lon1, lat1, lon2, lat2]:
        print("Invalid selection (clicked outside axes).")
        return

    min_lon, max_lon = sorted([lon1, lon2])
    min_lat, max_lat = sorted([lat1, lat2])

    # Store result
    selected_bbox = {
        "min_lon": round(min_lon, 6),
        "max_lon": round(max_lon, 6),
        "min_lat": round(min_lat, 6),
        "max_lat": round(max_lat, 6),
        "corners_lonlat": [
            [round(min_lon, 6), round(min_lat, 6)],  # SW
            [round(min_lon, 6), round(max_lat, 6)],  # NW
            [round(max_lon, 6), round(max_lat, 6)],  # NE
            [round(max_lon, 6), round(min_lat, 6)],  # SE
        ]
    }

    print("\nRectangle selected (WGS84):")
    for k, v in selected_bbox.items():
        print(f"{k}: {v}")
        

def toggle_selector(event):
    """Enable or disable selector with a key press."""
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print("RectangleSelector deactivated.")
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print("RectangleSelector activated.")
        toggle_selector.RS.set_active(True)

if True:
    # Set up map
    fig, ax = plt.subplots(
        figsize=(12, 6),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_title("Draw a rectangle (click and drag). Press 'a' to activate, 'q' to deactivate.")

    # Rectangle selector (API changed in Matplotlib >= 3.5 → no drawtype argument)
    toggle_selector.RS = RectangleSelector(
        ax, onselect,
        useblit=True,
        button=[1],  # left mouse button only
        minspanx=0, minspany=0,
        spancoords='data',
        interactive=True
    )

    plt.connect('key_press_event', toggle_selector)
    plt.show()

    # After window is closed
    if selected_bbox:
        print("\nFinal rectangle coordinates:")
        wgs84_data = selected_bbox["corners_lonlat"][0] + selected_bbox["corners_lonlat"][2]
        print(selected_bbox["corners_lonlat"][0] + selected_bbox["corners_lonlat"][2])

    else:
        print("\nNo rectangle selected.")



