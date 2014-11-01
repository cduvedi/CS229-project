import os
import csv
import numpy
import scipy
import random
from scipy import linalg
from scipy.misc import toimage

csv_file = open('/root/CS229-project/Datasets/train.csv', 'r');

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

index_sample_list = list()

for i in range(0, 7):
	if (len(index_list[i]) > 10):
		sample = random.sample(index_list[i], 10)
		for x_sample in sample:
			index_sample_list.append(x_sample)

random_face_list = list()
for i in index_sample_list:
	random_face_list.append(X_list[i])

random_face_array = numpy.array(random_face_list)

face_mean = numpy.mean(random_face_array, 0)
faces_adjusted = random_face_array - face_mean

faces_U, faces_S, faces_V = linalg.svd(faces_adjusted.transpose(), full_matrices=False)

feature_vector_list = list()
for i in range(0, len(X_list)):
	feature_vector_list.append(list())
print "Calculating eigen faces done"	

for i in range(0, len(X_list)):
	for U in faces_U.transpose():
		feature_vector_list[i].append(numpy.dot(X_list[i], U))

print "Mapped inputs to features"
feature_file = open('train.feat', 'w')
for X in feature_vector_list:
	 feature_file.write("%s\n" % X)

feature_file.close()

print "Feaute extraction done"
#toimage(numpy.reshape((faces_U.transpose()[1]), (48, 48))).show()
#toimage(numpy.reshape(X_list[i], (48, 48))).show()
