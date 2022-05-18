import os
import sys
import torch
import random
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn import metrics
import torch.nn.init as init
from tqdm import tqdm as tqdm
import torch.nn.functional as F
from sklearn.manifold import TSNE
from collections import defaultdict, OrderedDict

class Fully_Connected_Clustering_UCI(nn.Module):
	def __init__(self, args, writer):
		super().__init__()
		self.args = args
		self.writer = writer
		self.max_purity = 0
		self.max_RI = 0
		self.max_ARI = 0
		self.max_AMI = 0
		self.max_NMI = 0
		self.max_FMI = 0
		self.centroids = nn.Sequential(OrderedDict([
			("fc1", nn.Linear(self.args.num_features, self.args.num_clusters, bias = False)),
		]))
		nn.init.normal_(self.centroids.fc1.weight, mean = 0.0, std = 1.0)

	def forward(self, x):
		centroids = self.get_centroids()
		x_c_cdist = torch.cdist(x, centroids, p = self.args.distance_order)
		return x_c_cdist

	def get_pseudo_labels(self, x):
		centroids = self.get_centroids()
		x_c_cdist = torch.cdist(x, centroids, p = self.args.distance_order)
		_, pseudo_labels = torch.min(x_c_cdist, dim = 1)
		return pseudo_labels

	def get_centroids(self):
		return self.centroids.fc1.weight
	
	def get_bincounts(self, output):
		return torch.bincount(output, minlength = self.args.num_clusters)

	def centroids_sparsity(self):
		centroids = self.get_centroids()
		centroids_cdist = torch.cdist(centroids, centroids)
		return torch.sum(torch.triu(centroids_cdist))

	def fit(self, train_loader, criterion, optimizer, scheduler, num_epochs):
		for epoch in tqdm(range(num_epochs)):
			with torch.enable_grad():
				self.train()
				for i, (fnames, indexes, preprocessed_numerical_features, preprocessed_categorical_features, preprocessed_labels, _, _, _) in enumerate(train_loader):
					x = torch.cat((preprocessed_numerical_features, preprocessed_categorical_features), dim = 1).cuda(self.args.gpu)
					outputs = self.forward(x)
					probability = F.softmax(outputs, dim = 1)
					probability = 1 - probability
					predictions = probability.argmax(dim = 1)
					predictions_bincounts = self.get_bincounts(predictions)
					optimizer.zero_grad()
					loss = criterion(outputs, predictions)
					loss.backward()
					optimizer.step()
					purity = self.purity(preprocessed_labels, predictions.detach().cpu().numpy())
					RI = metrics.rand_score(preprocessed_labels, predictions.detach().cpu().numpy())
					ARI = metrics.adjusted_rand_score(preprocessed_labels, predictions.detach().cpu().numpy())
					AMI = metrics.adjusted_mutual_info_score(preprocessed_labels, predictions.detach().cpu().numpy())
					NMI = metrics.normalized_mutual_info_score(preprocessed_labels, predictions.detach().cpu().numpy())
					FMI = metrics.fowlkes_mallows_score(preprocessed_labels, predictions.detach().cpu().numpy())
					if purity > self.max_purity:
						self.max_purity = purity
					if RI > self.max_RI:
						self.max_RI = RI
					if ARI > self.max_ARI:
						self.max_ARI = ARI
					if AMI > self.max_AMI:
						self.max_AMI = AMI
					if AMI > self.max_NMI:
						self.max_NMI = NMI
					if FMI > self.max_FMI:
						self.max_FMI = FMI
				(self.writer).add_scalar("Loss", loss.item(), epoch)
				(self.writer).add_scalar("Purity", purity, epoch)
				(self.writer).add_scalar("Max Purity", self.max_purity, epoch)
				(self.writer).add_scalar("Max RI", self.max_RI, epoch)
				(self.writer).add_scalar("Max ARI", self.max_ARI, epoch)
				(self.writer).add_scalar("Max AMI", self.max_AMI, epoch)
				(self.writer).add_scalar("Max NMI", self.max_NMI, epoch)
				(self.writer).add_scalar("Max FMI", self.max_FMI, epoch)
				_ = [(self.writer).add_scalar("Bincount: {}".format(j), predictions_bincounts[j], epoch) for j in range(self.args.num_clusters)]
				scheduler.step()

	def purity(self, y_true, y_pred):
		contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
		return np.sum(np.amax(contingency_matrix, axis = 0)) / np.sum(contingency_matrix)

	def RI(self, y_true, y_pred):
		return metrics.rand_score(y_true, y_pred)
	

	def MI(self, y_true, y_pred):
		pass
