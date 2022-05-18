import os
import sys
import time
import json
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.util_loss import *
from lib.util_sampler import *
from lib.util_architecture import *

def set_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source_path",     							type = str,        default = "../dat/Heart_Disease/")
	parser.add_argument("--target_path",        						type = str,        default = "../log/")
	parser.add_argument("--env_file",        							type = str,        default = "Heart_Disease.json")
	parser.add_argument("--batch_size",    								type = int,   	   default = 1024)
	parser.add_argument("--num_epochs",    								type = int,        default = 200)
	parser.add_argument("--milestone_0",    							type = int,        default = 200)
	parser.add_argument("--milestone_1",    							type = int,        default = 200)
	parser.add_argument("--learning_rate",  					 		type = float,      default = 0.1)
	parser.add_argument("--num_clusters", 								type = int,        default = 2)
	parser.add_argument("--distance_order",         					type = int,        default = 2)
	parser.add_argument("--gpu",           								type = int,        default = 2)
	parser.add_argument("--random_seed",        						type = int,        default = 5)
	parser.add_argument("--model",              						type = str,        default = "Heart_Disease")
	parser.add_argument("--description",        						type = str,        default = "Fully Connected Clustering")
	return parser.parse_args()


if __name__ == "__main__":

	args = set_args()

	random.seed(args.random_seed)
	np.random.seed(args.random_seed)
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed(args.random_seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	print("source_path: {}".format(args.source_path))
	print("target_path: {}".format(args.target_path))
	print("env_file: {}".format(args.env_file))
	print("batch_size: {}".format(args.batch_size))
	print("num_epochs: {}".format(args.num_epochs))
	print("milestone_0: {}".format(args.milestone_0))
	print("milestone_1: {}".format(args.milestone_1))
	print("learning_rate: {}".format(args.learning_rate))
	print("num_clusters: {}".format(args.num_clusters))
	print("distance_order: {}".format(args.distance_order))
	print("gpu: {}".format(args.gpu))
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

	writer = SummaryWriter(log_dir = result_path)
	with open(os.path.join(result_path, "Experiment_Description.txt"), "w") as f:
	
		#################################################### Data Loader ###################################################

		dataset = XML_Dataset(os.path.join(args.source_path, "xml"))
		train_loader = DataLoader(dataset, shuffle = True, batch_size = args.batch_size)
		
		############################################## Experimental Settings ###############################################

		env = json.load(open(os.path.join(os.path.join(args.source_path, "env"), args.env_file), "r"))
		args.selected_features = env["selected_features"]
		args.numerical_features = env["numerical_features"]
		args.categorical_features = env["categorical_features"]
		args.label_feature = env["label_feature"]
		args.label_feature_mapping = env["label_feature_mapping"]
		args.num_features = len(args.numerical_features) + len(args.categorical_features)
		model = Fully_Connected_Clustering_UCI(args, writer).cuda(args.gpu)
		criterion = Clustering_Loss(args)
		optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [args.milestone_0, args.milestone_1], gamma = 0.1)

		################################################## Model Training ##################################################
			
		training_start_time = time.time()
		model.fit(train_loader, criterion, optimizer, scheduler, args.num_epochs)
		training_end_time = time.time()
		training_time = training_end_time - training_start_time
		print("Training time: {}".format(training_time))
		print("Max Purity: {}".format(model.max_purity))
		print("Max RI: {}".format(model.max_RI))
		print("Max ARI: {}".format(model.max_ARI))
		print("Max AMI: {}".format(model.max_AMI))
		print("Max NMI: {}".format(model.max_NMI))
		print("Max FMI: {}".format(model.max_FMI))

		############################################### Experiment Recording ###############################################
		
		end_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
		f.write("start_time: {}\n".format(start_time))
		f.write("end_time: {}\n".format(end_time))
		f.write("training_time: {}\n".format(training_time))
		f.write("source_path: {}\n".format(args.source_path))
		f.write("target_path: {}\n".format(args.target_path))
		f.write("env_file: {}\n".format(args.env_file))
		f.write("batch_size: {}\n".format(args.batch_size))
		f.write("num_epochs: {}\n".format(args.num_epochs))
		f.write("milestone_0: {}\n".format(args.milestone_0))
		f.write("milestone_1: {}\n".format(args.milestone_1))
		f.write("learning_rate: {}\n".format(args.learning_rate))
		f.write("num_clusters: {}\n".format(args.num_clusters))
		f.write("distance_order: {}\n".format(args.distance_order))
		f.write("max_purity: {}\n".format(model.max_purity))
		f.write("gpu: {}\n".format(args.gpu))
		f.write("random_seed: {}\n".format(args.random_seed))
		f.write("model: {}\n".format(args.model))
		f.write("description: {}\n".format(args.description))
		torch.save(model.state_dict(), os.path.join(os.path.join(os.path.join(args.target_path, args.model), start_time), "model.pth"))
