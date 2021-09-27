import time
import os

# Third party library
import pandas as pd
import numpy as np
from sklearn import metrics
from catboost import CatBoostClassifier, Pool, cv

# Our library
from config import *
from utils_func import *

def main():	
	# Load csv data
	print("Loading csv data")
	train_X1, train_y1, test_X1, test_y1 = load_csv_data(train_csv_joy, test_csv_joy)
	train_X2, train_y2, test_X2, test_y2 = load_csv_data(train_csv_handcraft, test_csv_handcraft)

	train_X = pd.concat([train_X1, train_X2], axis=1)
	test_X  = pd.concat([test_X1, test_X2], axis=1)
	train_y = train_y1
	test_y  = test_y1
	
	print("training sample number: " + str(len(train_y)))
	print("testing sample number : " + str(len(test_y)))
	print("-"*80)
	
	# Training
	print("Start training")
	start_time = time.time()
	model = catboost_train(train_X, train_y, test_X, test_y)
	end_time = time.time()
	print("time cost: " + str(int(end_time-start_time)) + " sec")
	print("-"*80)
	
	# Save weight file
	print("Save weight file in " + Weight_dir)
	weight_filename = "34_new_catboost_mixed.txt"
	model.save_model(Weight_dir + weight_filename)
	model.load_model(Weight_dir + weight_filename)
	print("-"*80)
	
	# Plot confusion matrix
	print("Plot confusion matrix in " + result_dir)
	predict_y = model.predict(test_X)
	filename  = result_dir + "34_new_catboost_mixed.png"
	predict_y = [labels[int(y)] for y in predict_y]
	test_y    = [labels[y] for y in test_y]
	cm_analysis(test_y, predict_y, filename, labels)
	print("-"*80)

	# Print the precision and recall, among other metrics
	print("Show experimental results: ")
	print(metrics.classification_report(test_y, predict_y, digits=8))
	print("-"*80)	
	
if __name__ == '__main__':
    main()
