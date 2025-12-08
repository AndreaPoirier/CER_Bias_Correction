import os, sys, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utilities.user_input import *
from Utilities.Functions.functions_plot import *
from Utilities.Functions.functions_statistics import *
from Utilities.Functions.functions_data_processing import *
from import_lib import *


############################################
#### EARTH ACCESS AUTHENTIFICATION  ########
############################################

# Login ONCE at the beginning
earthaccess.login(strategy="netrc")   

years = ["2024", "2025"]
months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
days = ["01", "15"]

#########################
#### OCI DATA  ##########
#########################

base_folder = r"D:\Data\OCI"
for year in years:
    for month in months:

        # Skip unwanted months for 2024
        if year == "2024" and int(month) < 7:
            continue

        # Skip unwanted months for 2025
        if year == "2025" and int(month) > 6:
            continue

        for day in days:

            current_day = f"{year}-{month}-{day}"
            time_tuple = (current_day, current_day)
            print("Searching:", time_tuple)

            # Search PACE OCI Cloud data
            results = earthaccess.search_data(
                short_name="PACE_OCI_L2_CLOUD",
                temporal=time_tuple
            )

            # Create correct output folder
            folder_path = os.path.join(base_folder, year, month, day)
            os.makedirs(folder_path, exist_ok=True)

            # Download files with authentication
            earthaccess.download(results, folder_path)

            print("Download done for:", folder_path)

print("\nDownloaded OCI data, now checking for corrupted files.\n")

base_folder_OCI = r"D:\Data\OCI"
removing_corrupted_files(base_folder_OCI)

print("\nDownloading HARP2 data ...\n ")    

###########################
#### HARP2 DATA  ##########
###########################

base_folder = "D:\\Data\\HARP2"
for year in years:
    for month in months:
        if year == "2024":
            if month == "01" or month == "02" or month == "03" or month == "04" or month == "05" or month == "06":
                continue
        if year == "2025":
            if month == "07" or month == "08" or month == "09" or month == "10" or month == "11" or month == "12":
                continue
        for day in days:
            current_day = year + "-" + month + '-' + day
            time_tuple = (current_day,current_day)
            print(time_tuple)

            results = earthaccess.search_data(
                short_name='PACE_HARP2_L2_CLOUD_GPC',               
                temporal=time_tuple, 
            )

            folder_path  = os.path.join(base_folder, year, month, day)

            # Ensure folder exists
            os.makedirs(folder_path, exist_ok=True)

            # Download files
            try:
                files_downloaded = earthaccess.download(results, folder_path)
                print("Download done for", folder_path)
            except:
                print("No data for", folder_path)

            
print("\nDownloaded HARP2 data, now checking for corrupted files.\n")


base_folder_HARP2 = r"D:\Data\HARP2"
removing_corrupted_files(base_folder_HARP2)