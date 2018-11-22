import os
from collections import Counter, OrderedDict
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
import pandas as pd
#from astropy.table import Table
import multiprocessing
#from cesium.time_series import TimeSeries
#import cesium.featurize as featurize
#from tqdm import tnrange, tqdm_notebook
import sklearn 
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, log_loss
import seaborn as sns

pd.set_option('display.max_columns', 500)

df = pd.read_csv("submission.csv")

#for index, row in df.iterrows():
#	row['object_id'] = row['object_id'].astype('int32')

#df['object_id'] = df['object_id'].values.astype(np.int64)

#df.drop('e', axis=1, inplace=True)

df = df.groupby(['object_id']).mean()

df.to_csv("submission.csv", index = False)