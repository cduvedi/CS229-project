#/usr/bin/python

import sys

IMAGE_ROW_SIZE = 48 
GABOR_FILTER_SIZE = IMAGE_ROW_SIZE * IMAGE_ROW_SIZE

def main():
	#dumpOutRandomFeatureFile(10) # For debug purposes

	num_args = len(sys.argv)
	if (num_args > 4):
		feature_imp_file = sys.argv[1]
		feature_vector_file = sys.argv[2]
		output_file = sys.argv[3]
		include_neighbor_pixels = bool(int(sys.argv[4]))
		print "Feature impotance file: ", feature_imp_file
		print "Dataset: ", feature_vector_file
		print "Output file:  ", output_file
		print "Neighours: ", include_neighbor_pixels

		sorted_feature_indices = getSortedFeatureIndices(feature_imp_file, include_neighbor_pixels)
		extract_selected_features(feature_vector_file, sorted_feature_indices, output_file)
	else:
		print("Invalid number of input args. Run as follows: <script> <feature_imp_file> <dataset> <output_features_file> <include_neighbor_pixels>")

def getSortedFeatureIndices(feature_imp_file, include_neighbor_pixels):
	feature_indices = set()
	f = open(feature_imp_file, 'r')
	for row in f:
		arr = row.split(",")
		idx = int(arr[0])
		imp = float(arr[1])
		if (imp > 0):
			feature_indices.add(idx)
			if (include_neighbor_pixels):
				for i in getNeighborPixelIndices(idx):
					feature_indices.add(i)

	f.close()
	lst = list(feature_indices)
	lst.sort()

	return lst

def extract_selected_features(feature_vector_file, selected_feature_indices, output_file):
	out_file = open(output_file, 'a')
	inp_file = open(feature_vector_file, 'r')

	for row in inp_file:
		label, feature_str = row.split(",")
		orig_features = feature_str.split(" ")
		selected_features = filter_features_by_index(orig_features, selected_feature_indices)
		s = reduce(lambda x,y : x + "," + y, selected_features)
		out_file.write(label + "," + s)

	inp_file.close()
	out_file.close()

def getNeighborPixelIndices(idx):
	n_indices = []
	gabor_filter_idx = int(idx / (GABOR_FILTER_SIZE))
	base_linear_addr = gabor_filter_idx * GABOR_FILTER_SIZE
	row_no = int((idx - base_linear_addr) / IMAGE_ROW_SIZE)
	col_no = idx - base_linear_addr - (row_no * IMAGE_ROW_SIZE)

	if (col_no > 0):
		n_indices.append(idx - 1)

	if (row_no > 0):
		n_indices.append(int(base_linear_addr + ((row_no - 1) * IMAGE_ROW_SIZE) + col_no))

	if (col_no < IMAGE_ROW_SIZE - 1):
		n_indices.append(idx + 1)

	if (row_no < IMAGE_ROW_SIZE - 1):
		n_indices.append(int(base_linear_addr + ((row_no + 1) * IMAGE_ROW_SIZE) + col_no))
	
	return n_indices

def filter_features_by_index(orig_features, selected_feature_indices):
	selected_features = []
	for idx in selected_feature_indices:
		selected_features.append(orig_features[idx])

	return selected_features

#=====================================
# Debug functions
#=====================================

def printArr(arr):
	print(reduce(lambda x,y : str(x) + "," + str(y), arr) + "\n")

def dumpOutRandomFeatureFile(num_samples):
	f = open('test/input_features.txt', 'w')

	for i in range(num_samples):
		vals = reduce(lambda x,y: str(x) + " " + str(y), map(lambda j: i*27 + j, range(27)))
		f.write(str(i) + "," + vals + "\n")

	f.close()


main()
