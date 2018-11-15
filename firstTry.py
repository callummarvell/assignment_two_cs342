# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 17:37:28 2018

@author: callu
"""

#%matplotlib inline

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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

pbmap = OrderedDict([(0,'u'), (1,'g'), (2,'r'), (3,'i'), (4, 'z'), (5, 'Y')])

# it also helps to have passbands associated with a color
pbcols = OrderedDict([(0,'blueviolet'), (1,'green'), (2,'red'),\
                      (3,'orange'), (4, 'black'), (5, 'brown')])

pbnames = list(pbmap.values())

#datadir = '../input/plasticc-astronomy-starter-kit-media'
metafilename = '../../../../../../courses/cs342/Assignment2/training_set_metadata.csv'
trainmeta = pd.read_csv(metafilename)

trainfilename = '../../../../../../courses/cs342/Assignment2/training_set.csv'
traindata = pd.read_csv(trainfilename)
print (traindata)

#testfilename = '../../../../../../courses/cs342/Assignment2/test_set.csv'
#testdata = pd.read_csv(testfilename)

#metafilename = '../../../../../../courses/cs342/Assignment2/test_set_metadata.csv'
#testmeta = pd.read_csv(metafilename)

#nobjects = len(trainmeta)
#print(metadata)

ts_lens = traindata.groupby(['object_id', 'passband']).size()
f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(ts_lens, ax=ax)
ax.set_title('distribution of time series lengths')

#obj_passband = traindata.groupby(['object_id'])
#f, ax = plt.subplots(figsize=(12, 6))
#sns.distplot(obj_passband['flux'], ax=ax)
#ax.set_title('distribution of flux values')

#f, ax = plt.subplots(figsize=(12, 6))
#sns.distplot(obj_passband['flux_err'], ax=ax)
#ax.set_title('distribution of flux error values')

f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(traindata['mjd'], ax=ax, bins=200)
ax.set_title('number of observations made at each time point')

f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(trainmeta['hostgal_specz'], ax=ax)
ax.set_title('distribution of spectroscopicic redshift of host galaxies')

f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(trainmeta['hostgal_photoz'], ax=ax)
ax.set_title('distribution of photometric redshift of host galaxies')

f, ax2 = plt.subplots(figsize=(12, 6))
ax2.scatter(x=trainmeta['ra'], y=trainmeta['decl'])
ax2.set_title('distribution of coordinates within sky using ra/decl system')
ax2.set_xlabel('ra')
ax2.set_ylabel('decl')

plt.show()