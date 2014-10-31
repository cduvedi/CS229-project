import os
import csv
import numpy

csv_file = open('/root/CS229-project/feature_extraction/train_small.csv', 'r');

reader = csv.reader(csv_file)

X_list = []
Y_list = []

for row in reader:
	Y_row, X_row_str = row
	X_row_split = X_row_str.split(' ')
	Y = int(Y_row)
	Y_list.append(Y)
	X_row = map(lambda x: float(x), X_row_split)
	X_list.append(X_row)

index_list = list()

for i in range(0, 7):
	index_list.append(set())

for (index, y) in enumerate(Y_list):
	index_list[y].add(index)

for i in range(0, 7):
	print len(index_list[i])
