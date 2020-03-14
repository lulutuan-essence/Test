import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


import seaborn as sb
import statsmodels.api as sm
import pylab as pl
from statsmodels.formula.api import logit

import seaborn as sns 
sns.set(style = 'white')
sns.set(style = 'whitegrid', color_codes = True)

pd.set_option('display.max_columns', 1500)
pd.set_option('display.max_rows', 700)
pd.set_option('display.max_colwidth', 200)
