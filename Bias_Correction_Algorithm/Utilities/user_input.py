import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from import_lib import *

####################################################
#                USER CONFIGURATION               
#  Edit the following variables to customize the 
#  input data, area of interest, temporal range, 
#  and analysis options.                          
####################################################


########## Data Storage Locations ##########

# Path to OCI L2 data
folder_path_OCI = r"D:\Data\OCI\2024\07\01"

# Path to HARP2 L2 data
folder_path_HARP2 = r"D:\Data\HARP2\2024\07\01"

# Path to HARP2 L2 data
folder_path_MODIS = r"D:\Data\MODIS\MYD06_L2_6.1-20251211_143824"

# Path to store results from get_processed_data.py
folder_path_saved_processed_data = r"Bias_Correction_Algorithm\output\processed_data"

# Path for output of regression and SHAP
folder_output = r"Bias_Correction_Algorithm\output\region"

# Path for output of SHAP
folder_SHAP = r"Bias_Correction_Algorithm\output\shap"

# Path for output of regression and SHAP
folder_train = r"Bias_Correction_Algorithm\output\regression"

########## Definition of Area of Interest #########

# Option 1
center_lat, center_lon = -50, -145
dx = 5
lat_min, lat_max = center_lat - dx, center_lat + dx
lon_min, lon_max = center_lon - dx, center_lon + dx 

# Option 2
world_data = False


########## Definition of Temporal Range ##########

# Option 1: Analyse all available time
day_data = True

# Option 2: Analyse n_days after start_day
n_days = 1
start_day = '20240701'  # Format: 'YYYYMMDD'

########## Definition of Other Analysis Options ##########

# Box width (in degrees) used for:
box_width_for_chi_and_SF_computation = 0.5

# Min number of points required to perform quantile mapping 
min_neighbors = 30

# Fraction of files used when getting MODIS, OCI and HARP2 data 
sample_fraction_files = 1

# Block size in Bias_Correction_Algorithm\processing\OCI.py
block_size = 30_000

# OCI chunk size
OCI_chunk_size = 300_000

# HARP2 chunk size
HARP2_chunk_size = 600_000

# MODIS chunk size
MODIS_chunk_size = 30_000

# Save to disk every N blocks 
save_every_n_blocks = 5 


# Variables for Bias_Correction_Algorithm\processing\OCI.py
overlap_rows = 0

leafsize_for_kdtree = 100

min_neighbors = globals().get("min_neighbors", 5)

box_width_for_chi_and_SF_computation = globals().get("box_width_for_chi_and_SF_computation", 0.5)

half = box_width_for_chi_and_SF_computation 

diag = np.sqrt(2) * half

