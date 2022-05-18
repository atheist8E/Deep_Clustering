import os
import torch
import torch.nn.functional as F
from xml.etree.ElementTree import parse


class XML_Dataset(torch.utils.data.Dataset):
	def __init__(self, xml_path):
		super().__init__()
		self.xml_path = xml_path
		self.xmls = sorted([x for x in os.listdir(xml_path) if '.xml' in x], key = lambda x: int(x[:-4]))

	def __len__(self):
		return len(os.listdir(self.xml_path))

	def __getitem__(self, idx):
		fname = self.xmls[idx]
		tree = parse(os.path.join(self.xml_path, fname))
		root = tree.getroot()

		index = int(root.find("Index").text)

		original_node = root.find("Original")
		original_numerical_node = original_node.find("Numerical")
		original_categorical_node = original_node.find("Categorical")
		original_labels = str(original_node.find("Label").text)

		original_numerical_features = [float(child.text) for child in original_numerical_node]
		original_categorical_features = [str(child.text) for child in original_categorical_node]

		preprocessed_node = root.find("Preprocessed")
		preprocessed_numerical_node = preprocessed_node.find("Numerical")
		preprocessed_categorical_node = preprocessed_node.find("Categorical")
		preprocessed_labels = int(preprocessed_node.find("Label").text)
		preprocessed_numerical_features = [float(child.text) for child in preprocessed_numerical_node]
		preprocessed_categorical_features = [float(child.text) for child in preprocessed_categorical_node]
		
		original_numerical_features = torch.tensor(original_numerical_features, dtype = torch.float)
		preprocessed_numerical_features = torch.tensor(preprocessed_numerical_features, dtype = torch.float)
		preprocessed_categorical_features = torch.tensor(preprocessed_categorical_features, dtype = torch.float)
		preprocessed_labels = torch.tensor(preprocessed_labels, dtype = torch.long)

		return fname, index, preprocessed_numerical_features, preprocessed_categorical_features, preprocessed_labels, original_numerical_features, original_categorical_features, original_labels
