import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


####################################################
#                USER CONFIGURATION               
#  Edit the following variables to customize the 
#  input data, area of interest, temporal range, 
#  and analysis options.                          
####################################################


########## Data Storage Locations ##########

# Path to OCI L2 data
folder_path_OCI = r"C:\Users\andreap\Documents\python_projects\Data\OCI\August_data\PACE_OCI_L2_CLOUD_3.1-20251112_085756"
# Path to HARP2 L2 data
folder_path_HARP2 = r"C:\Users\andreap\Documents\python_projects\Data\HARP2\PACE_HARP2_L2_CLOUD_GPC_3.0-20251014_091732"
# Path to cloud mask data from OCI
folder_path_cloud_mask = r"C:\Users\andreap\Documents\python_projects\Data\OCI\Cloud_mask\PACE_OCI_L2_CLOUD_MASK_3.1-20251030_081441"
# Path to store results from get_processed_data.py
#folder_path_saved_processed_data = r"C:\Users\andreap\Documents\Cloud_Effective_Radius_Bias_Correction_Algorithm\Bias_Correction_Algorithm\Data_Processing\Processed_Data\World_Month_6"
#folder_path_saved_processed_data = r"C:\Users\andreap\Documents\Cloud_Effective_Radius_Bias_Correction_Algorithm\Bias_Correction_Algorithm\Data_Processing\Processed_Data\Small"
folder_path_saved_processed_data = r"C:\Users\andreap\Documents\Cloud_Effective_Radius_Bias_Correction_Algorithm\Bias_Correction_Algorithm\Data_Processing\Processed_Data\World_data"
########## Definition of Area of Interest #########
folder_output = r"C:\Users\andreap\Documents\Cloud_Effective_Radius_Bias_Correction_Algorithm\Bias_Correction_Algorithm\Output\Regression_with_Cloud_Coverage"

# 3 options to define the area to be analysed
# Option 1: 
# - Area of Interest is a rectangle defined using the GUI interface by drawing a rectangle of world map
# - Un-comment following line: "from interactive_rectangle import wgs84_data" and set world_data = False
# Option 2: 
# - Area of Interest is a rectangle with center coordinates defined by center_lat and center_lon, a half width of dx 
# - Comment out following line: "from interactive_rectangle import wgs84_data" and set world_data = False
# Option 3:
# - Area of Interest is the world
# - Comment out following line: "from interactive_rectangle import wgs84_data" and set world_data = True

try:
    #from interactive_rectangle import wgs84_data
    lat_min_list = [wgs84_data[1]]
    lat_max_list = [wgs84_data[3]]
    lon_min_list = [wgs84_data[0]]
    lon_max_list = [wgs84_data[2]]
except:
    center_lat, center_lon = 45, -34
    center_lat, center_lon = 3.6, 26
    dx = 2.5
    lat_min, lat_max = center_lat - dx, center_lat + dx
    lon_min, lon_max = center_lon - dx, center_lon + dx 

world_data = True


########## Temporal Range ##########

# Option 1: Analyse all available time
# day_data = False

# Option 2: Analyse n_days after start_day
day_data = True
n_days = 1
start_day = '20250807'  # Format: 'YYYYMMDD'

########## Analysis Options ##########

# Include Lorg and OII parameters in analysis
# WARNING: Setting this to True can drastically increase runtime
use_Lorg_and_OII = False

# Box width (in degrees) used for:
# - chi computation (cloud homogeneity)
# - Scaling factor from quantile mapping
box_width_for_chi_and_SF_computation = 0.5

# Box width (in degrees) used for:
# - Lorg and OII computation
box_width_for_Lorg_and_OII_computation = 0.25

# Min number of points required to perform quantile mapping 
min_neighbors = 30

# Run code under Option 1 or Option 2 (see manual)
get_data_from_files = False

# Fraction of files used from folder path OCI
sample_fraction_files = 1

# Fraction of data outputted from get_processed_data.py
sample_fraction_get_processed_data_py = 0.99

# Process this many points at once
block_size = 100_000  

# Save to disk every N blocks
save_every_n_blocks = 5 

