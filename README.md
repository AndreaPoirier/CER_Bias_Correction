# Cloud Effective Radius Bias Correction Algorithm

## Introduction 

Accurate cloud effective radius (CER) retrievals are essential for understanding how clouds interact with radiation, which directly influences Earth’s energy balance and climate sensitivity. Even small biases in CER can propagate into large uncertainties in climate models and satellite-derived radiative forcing estimates, making improved accuracy critical for reliable climate prediction and atmospheric research. This algorithm aims to correct biases in sensors using the bispectral method, in which horizontal cloud inhomogeneity is typically ignored. The code was developed for NASA’s PACE OCI instrument, with the HARP2 instrument serving as a reference, though it is theoretically applicable to other OCI-like sensors. 

This algorithm was developed by Andrea Poirier during his internship (October 2025 – February 2026) in the Department of Earth Sciences at the Space Research Organisation Netherlands (SRON), under the supervision of Dr. Bastiaan van Diedenhoven.

Contact: A.Poirier@sron.nl or poirier.andrea@outlook.com

## User Manual for Algorithm 

The algorithm is intended to be flexible and used by other users. The file `user_input.py` gives many different options for the analysis. Here a description of each of them is given. 

<details>
<summary>  Data Storage Location</summary>

This algorithm makes use of the L2 Cloud Optical Properties (not NRT) of NASA's PACE OCI and HARP2.
The algorithm requires data to be saved locally. The data can be downloaded on NASA Search Data engine (url: https://search.earthdata.nasa.gov/).
In `user_input.py` change the following lines of code with the correct data location. 

```ruby
# Path to OCI L2 data
folder_path_OCI = r"C:\Users\andreap\Documents\python_projects\Data\OCI\August_data\PACE_OCI_L2_CLOUD_3.1-20251112_085756"
# Path to HARP2 L2 data
folder_path_HARP2 = r"C:\Users\andreap\Documents\python_projects\Data\HARP2\PACE_HARP2_L2_CLOUD_GPC_3.0-20251014_091732"
```

The code is designed to handle tens of millions of data point. To avoid RAM memory issues, the compuation is progressively save. The folder where this data is to be saved needs to be defined. 

```ruby
# Path to store results from get_processed_data.py
folder_path_saved_processed_data = r"C:\Users\andreap\Documents\GitHub\SRON\Bias_Correction_Algorithm\Data_Processing\Processed_Data\World_30"
```
</details>
<details>
<summary>  How is the data processed ?</summary>

The majority of the run time is due to the processing of the data which includes: downloading data from .nc files and computing scaling factors and cloud inhomogeneity parameters (chi, Lorg and OII). Therefore, there are two options in the code. The first one is to simply run the algorithm in one shot. Both the data processing and analysis (regression, neural network etc ...) will performed without the need of human intervention. The second option, which is highly recommended, is to divide the runs into two parts. The first one is the data processing, which then saves the processed data locally in .h5 files. The second run is the analysis. To activate option set 
```ruby
get_data_from_files = False
```
</details>

<details>
<summary>  Definition of Area of Interest </summary>

The algorithm allows for three options. Option 1 is to analyse the entire world. To activate this option change to:

```ruby
# from interactive_rectangle import wgs84_data"
world_data = True
```
Option 2 allows to define a regional area by defining a rectangle with center coordinates and half width of box defined by:

```ruby
except:
    center_lat, center_lon = -34, 9
    dx = 2.5
```

Option 3 allows to define a regional area by defining a rectangle with a GUI interface, to activate change to:

```ruby
from interactive_rectangle import wgs84_data 
world_data = False

```

</details>


<details>
<summary>  Definition of Temporal Range </summary>

For the temporal range, the algorithm allows for two options. Option 1 is to analyze data from all available times. To turn on this option change this:

```ruby
day_data = False
```
The second option perform the analysis for a specified number of days 'n_days' after the a certain date 'start_day' with format YYYYMMDD. To use this option change:
```ruby
day_data = True
n_days = 1
start_day = '20250809
```

</details>


<details>
<summary>  Other Analysis Options </summary>

There are various option the algorithm supports for the analysis. The first one is whether to use the parameter Lorg and OII (Biagioli & Tompkins, 2023) in the bias correction. Note this may increase the computation time significantly. To use these parameters in the analysis, change to:
```ruby
use_Lorg_and_OII = True
```
To compute the the scaling factor, cloud inhomogeneity parameter (chi) and Lorg/OII at a pixel, the algorithm requires a distributions of HARP2 CER, OCI Cloud Optical Thickness and OCI cloud mask respectively. The size (in degrees) of the square around the pixel of interest can be changed with this line:
```ruby
box_width_for_chi_and_SF_computation = 0.5
box_width_for_Lorg_and_OII_computation = 0.25
```
Additionally, to have a reasonable estimate of the aforementioned parameters, a minimum number of points inside the box is specified. If there are less data points than 'min_neighbors' then the code skips this pixel:

```ruby
min_neighbors = 30
```
As mentionned before, this algorithm is build with the intent of handling large datasets. There are two intermediate option of randomly sampling the data. 'sample_fraction_files' is the fraction of .nc files the code processes. 'sample_fraction_get_processed_data_py' is the fraction of datapoints that are handled before 'get_processed_data.py'.

```ruby
sample_fraction_files = 0.07
sample_fraction_get_processed_data_py = 1
```
To avoid RAM memory issues, 'get_processed_data.py' performs block wise computation. The size of the blocks can be changed. Additionally, to avoid loss of computation, intermediate saving is performed by the code every 'save_every_n_blocks'.

```ruby
block_size = 100_000  
save_every_n_blocks = 40  
```
</details>


## Required libraries 

- `cartopy`
- `datetime`
- `gc`
- `h5py`
- `itertools`
- `joblib`
- `lightgbm`
- `matplotlib`
- `mpl_toolkits`
- `netCDF4`
- `numba`
- `numpy`
- `pandas`
- `pyproj`
- `scipy`
- `sklearn`
- `torch`
- `tqdm`