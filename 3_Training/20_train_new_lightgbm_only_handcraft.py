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
	train_X, train_y, test_X, test_y = load_csv_data(train_csv_handcraft, test_csv_handcraft)
		
	print("training sample number: " + str(len(train_y)))
	print("testing sample number : " + str(len(test_y)))
	print("-"*80)
	
	# Training
	print("Start training")
	start_time = time.time()
	model = lgbm_train(train_X, train_y, test_X, test_y)
	end_time = time.time()
	print("time cost: " + str(int(end_time-start_time)) + " sec")
	print("-"*80)
	
	# Save weight file
	print("Save weight file in " + Weight_dir)
	weight_filename = "20_new_lightgbm_handcraft_only.txt"
	model.save_model(Weight_dir + weight_filename)
	model = lgb.Booster(model_file=Weight_dir + weight_filename)
	print("-"*80)
	
	# Plot confusion matrix
	print("Plot confusion matrix in " + result_dir)
	predict_y = model.predict(test_X, num_iteration = model.best_iteration)
	filename = result_dir + "20_new_lightgbm_handcraft_only.png"
	predict_y = [labels[y.argmax()] for y in predict_y]
	test_y    = [labels[y] for y in test_y]
	cm_analysis(test_y, predict_y, filename, labels)
	print("-"*80)

	# Print the precision and recall, among other metrics
	print("Show experimental results: ")
	print(metrics.classification_report(test_y, predict_y, digits=8))
	print("-"*80)
	
	# Plot the feature importance
	print("Plot feature importance map " + result_dir)
	ax = lgb.plot_importance(model, max_num_features=15, figsize=(15, 15))
	plt.savefig(result_dir + "20_new_lightgbm_handcraft_only_importance.png")
	print("-"*80)	
	
if __name__ == '__main__':
    main()
