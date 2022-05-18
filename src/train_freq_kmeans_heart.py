import os
import sys
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from datetime import datetime
from sklearn.cluster import KMeans

def set_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source_path",     							type = str,        default = "../dat/Heart_Disease/")
	parser.add_argument("--target_path",        						type = str,        default = "../log/")
	parser.add_argument("--env_file",        							type = str,        default = "Heart_Disease.json")
	parser.add_argument("--num_clusters", 								type = int,        default = 2)
	parser.add_argument("--random_seed",        						type = int,        default = 1)
	parser.add_argument("--model",              						type = str,        default = "Heart_Disease")
	parser.add_argument("--description",        						type = str,        default = "K Means & Frequency Encoding")
	return parser.parse_args()

def purity(y_true, y_pred):
	contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
	return np.sum(np.amax(contingency_matrix, axis = 0)) / np.sum(contingency_matrix)

if __name__ == "__main__":

	args = set_args()

	random.seed(args.random_seed)
	np.random.seed(args.random_seed)

	print("source_path: {}".format(args.source_path))
	print("target_path: {}".format(args.target_path))
	print("env_file: {}".format(args.env_file))
	print("num_clusters: {}".format(args.num_clusters))
	print("random_seed: {}".format(args.random_seed))
	print("model: {}".format(args.model))
	print("description: {}".format(args.description))

	start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

	model_path = os.path.join(args.target_path, args.model)
	if os.path.exists(model_path) is False:
		os.mkdir(model_path)
	result_path = os.path.join(model_path, start_time)

	if os.path.exists(result_path) is False:
		os.mkdir(result_path)
	args.result_path = result_path

	with open(os.path.join(result_path, "Experiment_Description.txt"), "w") as f:
	
		#################################################### Data Loader ###################################################

		dataset = pd.read_csv(os.path.join(os.path.join(args.source_path, "normalized"), "Heart_Disease.csv"))
		print(dataset)
		
		############################################## Experimental Settings ###############################################

		env = json.load(open(os.path.join(os.path.join(args.source_path, "env"), args.env_file), "r"))
		args.selected_features = env["selected_features"]
		args.numerical_features = env["numerical_features"]
		args.categorical_features = env["categorical_features"]
		args.label_feature = env["label_feature"]
		args.label_feature_mapping = env["label_feature_mapping"]
		args.num_features = len(args.numerical_features) + len(args.categorical_features)

		################################################## Model Training ##################################################
			
		training_start_time = time.time()
		model = KMeans(n_clusters = args.num_clusters, random_state = args.random_seed)
		predictions = model.fit_predict(dataset[args.numerical_features + args.categorical_features].to_numpy())
		training_end_time = time.time()
		training_time = training_end_time - training_start_time
		print("Training time: {}".format(training_time))
		print("Purity: {}".format(purity(dataset[args.label_feature].to_numpy(), predictions)))
		print("RI: {}".format(metrics.rand_score(dataset[args.label_feature].to_numpy(), predictions)))
		print("ARI: {}".format(metrics.adjusted_rand_score(dataset[args.label_feature].to_numpy(), predictions)))
		print("AMI: {}".format(metrics.adjusted_mutual_info_score(dataset[args.label_feature].to_numpy(), predictions)))
		print("NMI: {}".format(metrics.normalized_mutual_info_score(dataset[args.label_feature].to_numpy(), predictions)))
		print("FMI: {}".format(metrics.fowlkes_mallows_score(dataset[args.label_feature].to_numpy(), predictions)))

		############################################### Experiment Recording ###############################################
		
		end_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
		f.write("start_time: {}\n".format(start_time))
		f.write("end_time: {}\n".format(end_time))
		f.write("training_time: {}\n".format(training_time))
		f.write("source_path: {}\n".format(args.source_path))
		f.write("target_path: {}\n".format(args.target_path))
		f.write("env_file: {}\n".format(args.env_file))
		f.write("num_clusters: {}\n".format(args.num_clusters))
		f.write("random_seed: {}\n".format(args.random_seed))
		f.write("model: {}\n".format(args.model))
		f.write("description: {}\n".format(args.description))
