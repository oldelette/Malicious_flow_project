# Third party library
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from sklearn.svm import SVC

import lightgbm as lgb
from catboost import CatBoostClassifier, Pool, cv
from xgboost import XGBClassifier
import xgboost as xgb
from scipy import stats

# Our library
from config import *

""" csv data """
def load_csv_data(train_csv, test_csv):
	train_df = pd.read_csv(train_csv, sep='\t')
	test_df  = pd.read_csv(test_csv , sep='\t')	
	
	train_y = []
	for class_name in train_df["class"]:
		train_y.append(dic[class_name])		
	
	test_y = []
	for class_name in test_df["class"]:
		test_y.append(dic[class_name])
		
	train_X = train_df.drop(["Unnamed: 0", "class"], axis = 1)
	test_X  = test_df.drop(["Unnamed: 0", "class"], axis = 1)	
		
	return train_X, train_y, test_X, test_y

""" zeroshot csv data (only have 2 class, normal and malicious) """
def load_csv_data_zeroshot(train_csv, test_csv):
	train_df = pd.read_csv(train_csv, sep='\t')
	test_df  = pd.read_csv(test_csv , sep='\t')	
	
	train_y = []
	for class_name in train_df["class"]:
		if class_name == "normal":
			train_y.append(0)
		else:
			train_y.append(1)
	
	test_y = []
	for class_name in test_df["class"]:
		if class_name == "normal":
			test_y.append(0)
		else:
			test_y.append(1)
		
	train_X = train_df.drop(["Unnamed: 0", "class"], axis = 1)
	test_X  = test_df.drop(["Unnamed: 0", "class"], axis = 1)	
		
	return train_X, train_y, test_X, test_y

""" confusion matrix """
def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(12,12)):
	"""
	Generate matrix plot of confusion matrix with pretty annotations.
	The plot image is saved to disk.
	args: 
	  y_true:    true label of the data, with shape (nsamples,)
	  y_pred:    prediction of the data, with shape (nsamples,)
	  filename:  filename of figure file to save
	  labels:    string array, name the order of class labels in the confusion matrix.
		         use `clf.classes_` if using scikit-learn models.
		         with shape (nclass,).
	  ymap:      dict: any -> string, length == nclass.
		         if not None, map the labels & ys to more understandable strings.
		         Caution: original y_true, y_pred and labels must align.
	  figsize:   the size of the figure plotted.
	"""
	if ymap is not None:
		y_pred = [ymap[yi] for yi in y_pred]
		y_true = [ymap[yi] for yi in y_true]
		labels = [ymap[yi] for yi in labels]
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	cm_sum = np.sum(cm, axis=1, keepdims=True)
	cm_perc = cm / cm_sum.astype(float) * 100
	annot = np.empty_like(cm).astype(str)
	nrows, ncols = cm.shape	
	for i in range(nrows):
		for j in range(ncols):
		    c = cm[i, j]
		    p = cm_perc[i, j]
		    if i == j:
		        s = cm_sum[i]
		        annot[i, j] = '%.1f%%\n%d' % (p, c)
		    elif c == 0:
		        annot[i, j] = ''
		    else:
		        annot[i, j] = '%.1f%%\n%d' % (p, c)
	cm = pd.DataFrame(cm_perc, index=labels, columns=labels)
	cm.index.name = 'Actual'
	cm.columns.name = 'Predicted'
	fig, ax = plt.subplots(figsize=figsize)
	sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="Blues")
	plt.savefig(filename)

""" lightgbm """
def lgbm_train(train_X, train_y, test_X, test_y):

    lgb_train = lgb.Dataset(train_X, train_y)  
    lgb_eval = lgb.Dataset(test_X, test_y, reference=lgb_train) 
    
    # specify your configurations as a dict  
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': class_num,
		#'device': 'gpu',
        'metric': 'multi_error',    
        'num_leaves': 300,
        'max_depth': 10,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'min_data_in_leaf': 20,
        'bagging_freq': 5,
        'verbose': -1,
        #'is_unbalance':True,
        #'scale_pos_weight': 10
    }
       
    gbm = lgb.train(params,  
                    lgb_train,  
                    num_boost_round=1000,  
                    valid_sets=lgb_eval,                   
                    early_stopping_rounds=200)    
    
    return gbm


""" decision tree """
def decisionTree_train(train_X, train_y, test_X, test_y):	
	
	clf = DecisionTreeClassifier(class_weight="balanced")
	clf = clf.fit(train_X, train_y)
		
	return clf

""" random forest """
def RF_train(train_X, train_y, test_X, test_y):	
	
	clf = RandomForestClassifier(class_weight="balanced", n_estimators=100)
	clf = clf.fit(train_X, train_y)
		
	return clf

""" knn """
def knn_train(train_X, train_y, test_X, test_y):	
	
	clf = neighbors.KNeighborsClassifier(n_neighbors = 8)
	clf = clf.fit(train_X, train_y)
		
	return clf


""" catboost """
def catboost_train(train_X, train_y, test_X, test_y):      
    
    model = CatBoostClassifier(
    iterations=1000, 
    learning_rate=0.5, 
    depth=10,
    loss_function='MultiClass',
    custom_metric="Accuracy",
    eval_metric="Accuracy",
    #task_type="GPU",
    verbose = 1)
    # Fit model
    model.fit(train_X, train_y, eval_set=(test_X, test_y))
    
    return model
    
""" xgboost """
def xgboost_train(train_X, train_y, test_X, test_y):      
    
    xgb = XGBClassifier(booster='gbtree', colsample_bylevel=1,
                        colsample_bytree=0.7, gamma=0.1, learning_rate=0.5, max_delta_step=0,
                        max_depth=10, min_child_weight=3, missing=None, n_estimators=1000,
                        n_jobs=-1, objective='multi:softprob', num_class=class_num, #gpu_id=0, tree_method ='gpu_hist',
                        silent=0, subsample=0.7)    
   
    xgb.fit(train_X, train_y, eval_metric="merror")
    
    return xgb

""" svm """
def svm_train(train_X, train_y, test_X, test_y):   

    clf = SVC(gamma='auto')
    clf = clf.fit(train_X, train_y)
      
    return clf
   
""" zeroshot lightgbm """
def lgbm_train_zeroshot(train_X, train_y, test_X, test_y):

    lgb_train = lgb.Dataset(train_X, train_y)  
    lgb_eval = lgb.Dataset(test_X, test_y, reference=lgb_train) 
    
    # specify your configurations as a dict  
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 2,
		#'device': 'gpu',
        'metric': 'multi_error',    
        'num_leaves': 300,
        'max_depth': 10,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'min_data_in_leaf': 20,
        'bagging_freq': 5,
        'verbose': -1,
        #'is_unbalance':True,
        #'scale_pos_weight': 10
    }
       
    gbm = lgb.train(params,  
                    lgb_train,  
                    num_boost_round=1000,  
                    valid_sets=lgb_eval,                   
                    early_stopping_rounds=200)    
    
    return gbm

""" for analyzing feature """
def preprocess_data(train_X, train_y):
	feature_names = train_X.columns	
	# outliers to None
	for feature_name in train_X.columns:
		median = np.median(train_X[feature_name])
		std    = np.std(train_X[feature_name])
		mean   = np.mean(train_X[feature_name])
		
		for i in range(len(train_X)):
			value = train_X.iloc[i, train_X.columns.get_loc(feature_name)]
			if (value / median > 10e2) or (value / median < 10e-2) or (abs(value - mean) > 2 * std):
				train_X.iloc[i, train_X.columns.get_loc(feature_name)] = None
				
	# Normalization in range 0-bin_range
	scaler = MinMaxScaler((0, bin_range))	
	scaler.fit(train_X)
	train_X = scaler.transform(train_X)	
	train_X[np.isnan(train_X)] = -1	
	train_X = pd.DataFrame(train_X, columns=feature_names).astype("int64")
	
	train_X_normal    = []
	train_X_malicious = []
	
	for i, label in enumerate(train_y):
		features = train_X.iloc[i]
		
		if label == normal_label: # normal
			train_X_normal.append(features)
		else:
			train_X_malicious.append(features)			
	
	train_X_normal = pd.DataFrame(train_X_normal, columns = train_X.columns)
	train_X_malicious = pd.DataFrame(train_X_malicious, columns = train_X.columns)
		
	return train_X_normal, train_X_malicious
