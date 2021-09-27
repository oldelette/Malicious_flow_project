import os

# Third party library
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Our library
from config import *
from utils_func import *

def main():
	if os.path.isdir(analysis_dir) == False:
		os.mkdir(analysis_dir)
		
	train_X, train_y, test_X, test_y = load_csv_data(train_csv_joy, test_csv_joy)
	train_X_normal, train_X_malicious = preprocess_data(train_X, train_y)		
	
	# Sort feature importance
	model = lgb.Booster(model_file=Weight_dir + "10_new_lightgbm_joy_only.txt")
	feature_importance = pd.Series(model.feature_importance(), train_X.columns)
	feature_importance = feature_importance.sort_values(ascending=True)	
	not_important_feature_list = [feature_importance.index[i] for i in range(10)]
	
	for id, feature_name in enumerate(not_important_feature_list):
		# Preprocess data
		print("Preprocess feature: " + feature_name)
		normal_count = pd.value_counts(train_X_normal[feature_name])
		normal_count = normal_count.sort_index()
	
		malicious_count = pd.value_counts(train_X_malicious[feature_name])
		malicious_count = malicious_count.sort_index()
			
		bins = [i for i in range(bin_range)]
		normal_y = []
		malicious_y = []
	
		for i in range(bin_range):
			if i in list(normal_count.index):
				normal_y.append(normal_count.get(i))
			else:
				normal_y.append(0)
			
		for i in range(bin_range):
			if i in list(malicious_count.index):
				malicious_y.append(malicious_count.get(i))
			else:
				malicious_y.append(0)
		print("-"*80)		
		
		# Calculate duplicated rate
		duplicate_num = 0		
	
		for i in range(bin_range):
			duplicate_num += min(normal_y[i], malicious_y[i])
					
		try :
			duplicate_rate = float(duplicate_num) / min(sum(normal_y), sum(malicious_y))
		except:
			print("can't calculate feature: " + feature_name)
			continue
	
		print(feature_name + " duplicate rate: ", duplicate_rate)
		print("-"*80)
		
		# Plot feature distribution
		print("Plot feature distribution in " + analysis_dir)	
		fig = plt.figure(num=id, figsize=(16, 9))		
		plt.xlabel(feature_name + " value")
		plt.ylabel("number")
		plt.bar(bins, normal_y, alpha=1, color="orange",  label="normal") 
		plt.bar(bins, malicious_y, alpha=0.5, color="blue", label="malicious")
		#plt.hist([x, y], bins=bins, stacked=True)
		plt.legend(loc="upper right")		
		filename = analysis_dir + "41_" + feature_name + '.png' 
		plt.savefig(filename)
		print("-"*80)
	
if __name__ == '__main__':
    main()
