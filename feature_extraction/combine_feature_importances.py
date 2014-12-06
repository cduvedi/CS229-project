#!/usr/bin/python

import sys

def main():
	num_args = len(sys.argv)
	print("num_args: " + str(num_args))
	inp_files = []
	if (num_args > 1):
		for i in range(0,num_args - 1):
			inp_files.append(sys.argv[i + 1])

	combine_files(inp_files)

def combine_files(inp_files):
	print("Combining the following feature importance files")
	for f in inp_files:
		print(f)

	feature_to_imp = {}
	for inp_file in inp_files:
		f = open(inp_file, 'r')
		for row in f:
			arr = row.split(",")
			idx = int(arr[0])
			imp = float(arr[1])
			if (not(idx in feature_to_imp)):
				feature_to_imp[idx] = 0

			feature_to_imp[idx] = feature_to_imp[idx] + imp

		f.close()

	sorted_feature_idx_imp_pairs = sorted(feature_to_imp.items(), key=lambda (k,v) : -1*v)
	out_file = open('combined_feat_importances.txt', 'w')
	for idx, imp in sorted_feature_idx_imp_pairs:
		out_file.write(str(idx) + "," + str(imp) + "\n")
	out_file.close()

main()
