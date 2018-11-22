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
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.neural_network import MLPClassifier
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
    
    
    aggdata = data.groupby(['object_id']).agg({'mjd': ['min', 'max', 'size'],
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
column_data = list(traindata)


testfilename = '../../../../../../courses/cs342/Assignment2/test_set.csv'
#testdata = pd.read_csv(testfilename)

chunksize = 10 ** 6

#testdata = pd.DataFrame(columns=column_names)

testdata = pd.read_csv(testfilename, nrows=chunksize)

#for chunk in pd.read_csv(testfilename, chunksize=chunksize):
#    testdata = testdata.append(chunk)
#    break

column_meta = list(trainmeta.drop('target',axis=1))

print(column_meta)

metafilename = '../../../../../../courses/cs342/Assignment2/test_set_metadata.csv'
#testmeta = pd.DataFrame(columns=column_names)

#for chunk in pd.read_csv(metafilename, chunksize=chunksize):
#    testmeta = testmeta.append(chunk)
#    break

testmeta = pd.read_csv(metafilename, nrows=chunksize)
    
#nobjects = len(trainmeta)
#print(metadata)

print(testdata['flux'][:10])
print("Cheese")

pret = get_inputs(testdata, testmeta)


pre = get_inputs(traindata, trainmeta)

print(pre[:5])
pre['hostgal_specz'].fillna(9999, inplace=True)
pre['distmod'].fillna(9999, inplace=True)

pre = pre.dropna()

X = np.array(pre.drop('target', axis=1).iloc[:,:])
yp = pre['target']
y = np.array(pre['target']).ravel()

#classes = sorted(yp.unique())

class_weights = {
    c:1 for c in classes
}

for c in [64, 15]:
    class_weights[c] = 2
    
print(class_weights)

print(X[5:15])
print(pre.iloc[5:30,:])
print("+++++++++++")
print(y)

print(pre.isnull().any())

clf = MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100, 100), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

clf.classes_=classes
#skf = StratifiedKFold(n_splits=2)

"""
parameters = {'solver':('lbfgs','sgd','adam'), 'activation':('identity','logistic','tanh','relu'), 'learning_rate':('constant','invscaling','adaptive'), 'hidden_layer_sizes':((100,),(100,100,),(100,100,100,))}
gcv = GridSearchCV(estimator=clf, param_grid=parameters, cv=10, verbose=100, n_jobs=-1)

gcv.fit(X,y)

best = gcv.best_estimator_
print(best)
print(gcv.best_score_)
print(gcv.best_params_)
"""
"""
for train_index, test_index in skf.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	
	clf.fit(X_train, y_train)
	proba = clf.predict_proba(X_test)
	print(proba[:20])
	print(clf.score(X_test, y_test))
	print(log_loss(y_test, proba))
"""

clf.fit(X,y)

skip = 0
cols = ["class_6", "class_15","class_16", "class_42","class_52", "class_53","class_62", "class_64","class_65", "class_67","class_88", "class_90","class_92", "class_95"]
testmeta = pd.read_csv(metafilename)
"""
testdata = pd.read_csv(testfilename, skiprows=skip, nrows=chunksize, header=0, names=column_data)
#print("TESTING")
#print(testdata[:5])
#print(testmeta[:5])
skip += chunksize
pre_test = get_inputs(testdata, testmeta)
#pre_test = pre_test.dropna()
print(pre_test.iloc[:,:10])
print(pre_test.isnull().any())
pre_test.fillna(0, inplace=True)
Xt = np.array(pre_test.iloc[:,:])
out = clf.predict_proba(Xt)
out2 = clf.predict(Xt)
try:
	new_df = pd.DataFrame(out, columns=cols)
	new_df['e'] = pd.Series(np.array(pre_test['object_id']).ravel(), index=new_df.index)
	df = pd.concat([df, new_df])
	new_df = pd.DataFrame(out2)
	df2 = pd.concat([df2, new_df])
except NameError:
	df = pd.DataFrame(out, columns=cols)
	df['e'] = pd.Series(np.array(pre_test['object_id']).ravel(), index=df.index)
	df2 = pd.DataFrame(out2)
"""

for chunk in pd.read_csv(testfilename, chunksize=chunksize):
	testdata = pd.read_csv(testfilename, skiprows=skip, nrows=chunksize, header=0, names=column_data)
	print(skip)
	#print("TESTING")
	#print(testdata[:5])
	#print(testmeta[:5])
	skip += chunksize
	pre_test = get_inputs(testdata, testmeta)
	#pre_test = pre_test.dropna()
	#print(pre_test)
	pre_test.fillna(9999, inplace=True)
	Xt = np.array(pre_test.iloc[:,:])
	out = clf.predict_proba(Xt)
	out2 = clf.predict(Xt)
	try:
		new_df = pd.DataFrame(out, columns=cols)
		new_df['object_id'] = pd.Series(np.array(pre_test['object_id']).ravel(), index=new_df.index)
		df = pd.concat([df, new_df])
		#new_df = pd.DataFrame(out2)
		#df2 = pd.concat([df2, new_df])
	except NameError:
		df = pd.DataFrame(out, columns=cols)
		df['object_id'] = pd.Series(np.array(pre_test['object_id']).ravel(), index=df.index)
		#df2 = pd.DataFrame(out2)

cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
df["class_99"] = '0.0'
df['object_id'] = df['object_id'].values.astype(np.int64)
print(df.iloc[:,:10])
df.to_csv("submission.csv", index = False)
#df2.to_csv("submission2.csv", index = False)
#parameters = {'criterion':('gini','entropy'), 'max_features':('auto','log2',None)}
#gcv = GridSearchCV(estimator=clf, param_grid=parameters, cv=10)

#gcv.fit(X,y)

#best = gcv.best_estimator_
#print(best)
#print(gcv.best_score_)
#print(gcv.best_params_)
"""
clf.fit(X, y)

pre_test = get_inputs(testdata, testmeta)

pre_test = pre_test.dropna()

Xt = np.array(pre_test.iloc[:,:])

print(Xt[5:15])
print(pre_test.iloc[5:30,:])


yt = np.array(pre_test['target']).ravel()



print(clf.score(Xt,yt))

proba = clf.predict_proba(Xt)

print(log_loss(yt, proba))
"""
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


#MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
#       beta_2=0.999, early_stopping=False, epsilon=1e-08,
#       hidden_layer_sizes=(100, 100), learning_rate='constant',
#       learning_rate_init=0.001, max_iter=200, momentum=0.9,
#       nesterovs_momentum=True, power_t=0.5, random_state=None,
#       shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1,
#       verbose=False, warm_start=False)

