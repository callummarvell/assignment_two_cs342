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
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

pd.set_option('display.max_columns', 500)

def get_inputs(data, metadata):
    data['flux_ratio_sq'] = np.power(data['flux'] / data['flux_err'], 2.0)
    data['flux_by_flux_ratio_sq'] = data['flux'] * data['flux_ratio_sq']
    mjddelta = data[['object_id','detected','mjd']]
    mjddelta = mjddelta[mjddelta.detected==1].groupby('object_id').agg({'mjd': ['min', 'max']})
    mjddelta['delta'] = mjddelta[mjddelta.columns[1]]-mjddelta[mjddelta.columns[0]]
    mjddelta = mjddelta['delta'].reset_index(drop=False)
    metadata = metadata.merge(mjddelta,on='object_id')
    
    
    aggdata = data.groupby(['object_id','passband']).agg({'mjd': ['min', 'max', 'size'],
                                             'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
                                             'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
                                             'flux_by_flux_ratio_sq': ['sum'],    
                                             'flux_ratio_sq': ['sum'],                      
                                             'detected': ['mean','std']}).reset_index(drop=False)
    
    cols = ['_'.join(str(s).strip() for s in col if s) if len(col)==2 else col for col in aggdata.columns ]
    aggdata.columns = cols
    aggdata = aggdata.merge(metadata,on='object_id',how='left')
    aggdata.insert(1,'delta_passband', aggdata.mjd_max-aggdata.mjd_min)
    aggdata.drop(['mjd_min','mjd_max'],inplace=True,axis=1)
    aggdata['flux_diff'] = aggdata['flux_max'] - aggdata['flux_min']
    aggdata['flux_dif2'] = (aggdata['flux_max'] - aggdata['flux_min']) / aggdata['flux_mean']
    aggdata['flux_w_mean'] = aggdata['flux_by_flux_ratio_sq_sum'] / aggdata['flux_ratio_sq_sum']
    aggdata['flux_dif3'] = (aggdata['flux_max'] - aggdata['flux_min']) / aggdata['flux_w_mean']
    return aggdata

classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99]

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
#print (traindata)

#testfilename = '../../../../../../courses/cs342/Assignment2/test_set.csv'
#testdata = pd.read_csv(testfilename)

#metafilename = '../../../../../../courses/cs342/Assignment2/test_set_metadata.csv'
#testmeta = pd.read_csv(metafilename)

#nobjects = len(trainmeta)
#print(metadata)



pre = get_inputs(traindata, trainmeta)

X = np.array(pre.drop(['distmod', 'target', 'flux_skew', 'flux_err_skew'], axis=1).iloc[:,:])
y = np.array(pre['target']).ravel()

print(X[5:15])
print(pre.iloc[5:30,:])
print("+++++++++++")
print(y)

print(pre.isnull().any())
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
parameters = {'criterion':('gini','entropy'), 'max_features':('auto','log2',None)}
gcv = GridSearchCV(estimator=clf, param_grid=parameters, cv=10)

gcv.fit(X,y)

best = gcv.best_estimator_
print(best)
print(gcv.best_score_)
print(gcv.best_params_)

"""
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
"""

#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
#            max_depth=None, max_features='log2', max_leaf_nodes=None,
#            min_impurity_split=1e-07, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            n_estimators=100, n_jobs=-1, oob_score=False,
#            random_state=None, verbose=0, warm_start=False)
#0.632390417941
#{'max_features': 'log2', 'criterion': 'entropy'}


