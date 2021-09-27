import time
import os

# Third party library
import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Our library
from config import *
from utils_func import *

def main():	
	# Load csv data
	print("Loading csv data")
	train_X, train_y, test_X, test_y = load_csv_data(train_csv_handcraft, test_csv_handcraft)
	
	# Normalization 
	scaler = StandardScaler()
	scaler.fit(train_X)
	train_X = scaler.transform(train_X)
	test_X  = scaler.transform(test_X)	
	
	print("training sample number: " + str(len(train_y)))
	print("testing sample number : " + str(len(test_y)))
	print("-"*80)
	
	# Training
	print("Start training")
	start_time = time.time()
	model = RF_train(train_X, train_y, test_X, test_y)
	end_time = time.time()
	print("time cost: " + str(int(end_time-start_time)) + " sec")
	print("-"*80)
	
	# Save weight file
	print("Save weight file in " + Weight_dir)
	weight_filename = "21_new_rf_handcraft_only.pickle"

	with open(Weight_dir+weight_filename, 'wb') as f:
		pickle.dump(model, f)

	with open(Weight_dir+weight_filename, 'rb') as f:
		model = pickle.load(f)
	print("-"*80)
	
	# Plot confusion matrix
	print("Plot confusion matrix in " + result_dir)
	predict_y = model.predict(test_X)
	filename = result_dir + "21_new_rf_handcraft_only.png"
	predict_y = [labels[y] for y in predict_y]
	test_y    = [labels[y] for y in test_y]
	cm_analysis(test_y, predict_y, filename, labels)
	print("-"*80)

	# Print the precision and recall, among other metrics
	print("Show experimental results: ")
	print(metrics.classification_report(test_y, predict_y, digits=8))
	print("-"*80)	

if __name__ == '__main__':
    main()
