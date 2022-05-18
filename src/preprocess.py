import os
import sys
import json
import time
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, Process
from sklearn.preprocessing import StandardScaler
from xml.etree.ElementTree import Element, SubElement, ElementTree, parse

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", 									type = str, 		default = "../dat/Heart_Disease/")
    parser.add_argument("--source_file", 									type = str, 		default = "Heart_Disease.csv")
    parser.add_argument("--env_file", 										type = str, 		default = "Heart_Disease.json")
    parser.add_argument("--num_workers", 									type = str, 		default = 10)
    return parser.parse_args()

def imputation():
	global args
	df = pd.read_csv(os.path.join(os.path.join(args.source_path, "raw"), args.source_file))
	if not os.path.exists(os.path.join(args.source_path, "imputed")):
		os.mkdir(os.path.join(args.source_path, "imputed"))
	df.dropna(axis = 0, inplace = True)
	df.to_csv(os.path.join(os.path.join(args.source_path, "imputed"), args.source_file), index = False)

def frequency_encoding():
	global args
	df = pd.read_csv(os.path.join(os.path.join(args.source_path, "imputed"), args.source_file))
	if not os.path.exists(os.path.join(args.source_path, "encodded")):
		os.mkdir(os.path.join(args.source_path, "encodded"))
	for feature in args.categorical_features:
		stat_feature_wise_categorical = df.groupby(feature).size()/len(df)
		df.loc[:, feature] = df[feature].map(stat_feature_wise_categorical)
	df.to_csv(os.path.join(os.path.join(args.source_path, "encodded"), args.source_file), index = False)

def label_encoding():
	global args
	df = pd.read_csv(os.path.join(os.path.join(args.source_path, "encodded"), args.source_file))
	df = df.astype({args.label_feature: str})
	df.loc[:, args.label_feature] = df[args.label_feature].map(args.label_feature_mapping)
	df.to_csv(os.path.join(os.path.join(args.source_path, "encodded"), args.source_file), index = False)

def standardization():
	global args
	if not os.path.exists(os.path.join(args.source_path, "normalized")):
		os.mkdir(os.path.join(args.source_path, "normalized"))
	df = pd.read_csv(os.path.join(os.path.join(args.source_path, "encodded"), args.source_file))
	df_X = df.drop([args.label_feature], axis = 1)
	df_y = df[args.label_feature]
	std_scaler = StandardScaler()
	df_X = pd.DataFrame(std_scaler.fit_transform(df_X), columns = df_X.columns)
	df = pd.concat([df_X, df_y], axis = 1)
	df.to_csv(os.path.join(os.path.join(args.source_path, "normalized"), args.source_file), index = False)

def csv_to_xml(zipped_row):
	global args
	original, preprocessed = zipped_row
	original_index, original_row = original
	preprocessed_index, preprocessed_row = preprocessed
	root = Element("XML")
	SubElement(root, "Index").text = str(original_index)
	original_node = SubElement(root, "Original")
	original_numerical_node = SubElement(original_node, "Numerical")
	original_categorical_node = SubElement(original_node, "Categorical")
	preprocessed_node = SubElement(root, "Preprocessed")
	preprocessed_numerical_node = SubElement(preprocessed_node, "Numerical")
	preprocessed_categorical_node = SubElement(preprocessed_node, "Categorical")

	for key, value in original_row.items():
		if key in args.selected_features:
			if key == args.label_feature:
				SubElement(original_node, "Label").text = str(value)
			else:
				if key in args.categorical_features:
					SubElement(original_categorical_node, key).text = str(value)
				else:
					SubElement(original_numerical_node, key).text = str(value)

	for key, value in preprocessed_row.items():
		if key in args.selected_features:
			if key == args.label_feature:
				SubElement(preprocessed_node, "Label").text = str(int(value))
			else:
				if key in args.categorical_features:
					SubElement(preprocessed_categorical_node, key).text = str(value)
				else:
					SubElement(preprocessed_numerical_node, key).text = str(value)

	tree = ElementTree(root)
	tree.write(os.path.join(os.path.join(args.source_path, "xml"), str(original_index) + ".xml"))
	
if __name__ == "__main__":
	args = set_args()
	env = json.load(open(os.path.join(os.path.join(args.source_path, "env"), args.env_file), "r"))
	args.selected_features = env["selected_features"]
	args.numerical_features = env["numerical_features"]			
	args.categorical_features = env["categorical_features"]			
	args.label_feature = env["label_feature"]
	args.label_feature_mapping = env["label_feature_mapping"]
	
	start = time.time()
	
	imputation()

	frequency_encoding()

	label_encoding()

	standardization()

	if not os.path.exists(os.path.join(args.source_path, "xml")):
		os.mkdir(os.path.join(args.source_path, "xml"))
	original_df = pd.read_csv(os.path.join(os.path.join(args.source_path, "raw"), args.source_file))
	preprocessed_df = pd.read_csv(os.path.join(os.path.join(args.source_path, "normalized"), args.source_file))
	with Pool(args.num_workers) as p:
		p.map(csv_to_xml, zip(original_df.iterrows(), preprocessed_df.iterrows()))
	
	print("execution time: {}".format(time.time() - start))
