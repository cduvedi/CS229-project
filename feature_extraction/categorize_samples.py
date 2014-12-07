#!/usr/bin/python

import os
import sys

NEGATIVE = 0
POSITIVE = 1

ANGER = 0
DISGUST = 1
FEAR = 2
SAD = 4

HAPPY = 3
SURPRISED = 5

NEUTRAL = 6

CATEGORIES = {
	ANGER 	  : NEGATIVE,
	FEAR      : NEGATIVE,
	DISGUST   : NEGATIVE,
	SAD  	  : NEGATIVE,
	HAPPY 	  : POSITIVE,
	SURPRISED : POSITIVE,
	NEUTRAL   : POSITIVE

}

def main():
	num_args = len(sys.argv)
	if (num_args > 1):
		feature_vector_file = sys.argv[1]
	
		print("dataset: " + feature_vector_file + "\n")

		categorize_dataset(feature_vector_file)

	else:
		print("Invalid number of input args.")
		print("Run the script as follows: ")
		print("    <script> <dataset>")

def categorize_dataset(feature_vector_file):
	inp_file_name = os.path.basename(feature_vector_file).split(".")[0]
	positive_emotions_file = open(inp_file_name + "_positive.csv", 'w')
	negative_emotions_file = open(inp_file_name + "_negative.csv", 'w')
	mapped_labels_file = open(inp_file_name + "_categorized.csv", 'w')
	f = open(feature_vector_file, 'r')

	for row in f:
		arr = row.split(",")
		label = int(arr[0])
		label_category = CATEGORIES[label]
		mapped_labels_file.write(str(label_category) + "," + arr[1] + "\n")
		if (label_category == POSITIVE):
			positive_emotions_file.write(row)
		else:
			negative_emotions_file.write(row)

	f.close()
	mapped_labels_file.close()
	negative_emotions_file.close()
	positive_emotions_file.close()

main()
