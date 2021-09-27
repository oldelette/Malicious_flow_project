import time
import os

# Third party library
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics

# Our library
from config import *
from utils_func import *

def main():	
	# Load csv data
	print("Loading csv data")
	train_X1, train_y1, test_X1, test_y1 = load_csv_data_zeroshot(train_csv_joy, zeroshot_csv_joy)
	train_X2, train_y2, test_X2, test_y2 = load_csv_data_zeroshot(train_csv_handcraft, zeroshot_csv_handcraft)
	
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
	model = lgbm_train_zeroshot(train_X, train_y, test_X, test_y)
	end_time = time.time()
	print("time cost: " + str(int(end_time-start_time)) + " sec")
	print("-"*80)
	
	# Save weight file
	print("Save weight file in " + Weight_dir)
	weight_filename = "52_new_lightgbm_mixed_zeroshot.txt"
	model.save_model(Weight_dir + weight_filename)
	model = lgb.Booster(model_file=Weight_dir + weight_filename)
	print("-"*80)
	
	# Plot confusion matrix
	print("Plot confusion matrix in " + result_dir)
	predict_y = model.predict(test_X, num_iteration = model.best_iteration)
	filename = result_dir + "52_new_lightgbm_mixed_zeroshot.png"
	predict_y = [binary_labels[y.argmax()] for y in predict_y]
	test_y    = [binary_labels[y] for y in test_y]
	cm_analysis(test_y, predict_y, filename, binary_labels)
	print("-"*80)

	# Print the precision and recall, among other metrics
	print("Show experimental results: ")
	print(metrics.classification_report(test_y, predict_y, digits=8))
	print("-"*80)
	
	# Plot the feature importance
	print("Plot feature importance map " + result_dir)
	ax = lgb.plot_importance(model, max_num_features=15, figsize=(15, 15))
	plt.savefig(result_dir + "52_new_lightgbm_mixed_importance_zeroshot.png")
	print("-"*80)	
	
if __name__ == '__main__':
    main()
