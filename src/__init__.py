# basic imports
import pandas as pd
import numpy as np
import time
from math import sqrt
import pickle

import warnings
with warnings.catch_warnings(): warnings.filterwarnings("ignore")

# model imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVR


# sklearn support
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization

# visualization
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

# metrics import
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer, r2_score
from eli5.sklearn import PermutationImportance
from eli5 import explain_weights_df




temporal = ['dayofyear', 'year', 'weekdays_Friday', 'weekdays_Monday',
       'weekdays_Saturday', 'weekdays_Sunday', 'weekdays_Thursday',
       'weekdays_Tuesday', 'weekdays_Wednesday', 'months_Apr', 'months_Aug',
       'months_Dec', 'months_Feb', 'months_Jan', 'months_Jul', 'months_Jun',
       'months_Mar', 'months_May', 'months_Nov', 'months_Oct', 'months_Sep',
       'season_fall', 'season_spring', 'season_summer', 'season_winter']


gases = ['D_NO', 'D_NO2', 'D_PM10', 'N_NO', 'N_NO2', 'N_Ox', 'N_O3', 'N_PM10',
         'O_NO', 'O_NO2', 'O_PM10', 'S_NO', 'S_NO2', 'S_Ox', 'S_O3', 'S_PM10',
          'W_NO', 'W_NO2', 'W_PM10']