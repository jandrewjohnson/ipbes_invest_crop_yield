import gdal

import pandas as pd
import numpy as np
from scipy import stats

 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import xgboost as xgb
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import permutation_test_score

import hazelbean as hb

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib 
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.basemap import Basemap
#%matplotlib inline

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


import pickle

def ff(e):
	print(e)