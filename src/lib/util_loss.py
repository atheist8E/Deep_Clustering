import torch
import torch.nn as nn
import torch.nn.functional as F

class Clustering_Loss(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args

	def forward(self, outputs, labels):
		probs = F.softmax(outputs, dim = 1)
		probs = 1 - probs
		max_probs, max_index = torch.max(probs, dim = 1)
		prediction_bincounts = torch.bincount(max_index)
		sample_weights = max_probs
		cluster_weights = (1 - prediction_bincounts / torch.sum(prediction_bincounts).type(torch.float64))
		log_probs = probs.log()
		loss_per_sample = F.nll_loss(log_probs, labels, reduction = "none")
		loss_weighted = loss_per_sample * sample_weights * cluster_weights[max_index]
		loss = torch.mean(loss_weighted)
		return loss
