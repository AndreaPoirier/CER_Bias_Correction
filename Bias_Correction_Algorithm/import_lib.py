import cartopy.crs as ccrs
import cartopy.feature as cfeature

from collections import defaultdict

from datetime import datetime, timedelta

import earthaccess

import gc

import h5py

import itertools
import time
from joblib import Parallel, delayed
from pyhdf.SD import SD, SDC

from lightgbm.callback import CallbackEnv
import lightgbm as lgb

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.widgets import RectangleSelector
from matplotlib.collections import PolyCollection
import matplotlib.gridspec as gridspec

from mpl_toolkits.mplot3d import Axes3D

from netCDF4 import Dataset


import numba as nb

import numpy as np

import pandas as pd

from pyproj import Proj, Transformer
from pyproj import Geod

from scipy.interpolate import griddata
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_2d
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from scipy.stats import pearsonr

from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import lognorm

import shap

from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import xarray as xr



