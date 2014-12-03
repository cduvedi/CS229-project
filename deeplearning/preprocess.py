#!/usr/bin/env python
# preprocess.py - Preprocess the data

# Common imports
import os
import sys
import time
import numpy
import csv

from sklearn.preprocessing import normalize

# Imorts from other custom modules
def load_set(file):
	X = list()
	Y = list()
	
	filehandle = open(file, 'r')
	reader = csv.reader(filehandle)
	
	for row in reader:
		labelStr, featureStr, tp = row
		label = int(labelStr)
		features = map(lambda x: float(x), featureStr.split(' '))
		X.append(features)
		Y.append(label)
	
	xy = [X, Y]

	return(xy)

def preProcessImages(datasets):
	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

	# For each training image, validation image, test image
	#	Subtract the mean of the image from each pixel
	#	Normalize the norm to 10
	for idx, image in enumerate(train_set_x):
		img_mean = numpy.mean(image)	
		for idy, pixel in enumerate(image):
			train_set_x[idx][idy] = train_set_x[idx][idy] - img_mean
	
	train_set_x = normalize(train_set_x, axis=1) * 100

	for idx, image in enumerate(valid_set_x):
		img_mean = numpy.mean(image)	
		for idy, pixel in enumerate(image):
			valid_set_x[idx][idy] = valid_set_x[idx][idy] - img_mean
	
	valid_set_x = normalize(valid_set_x, axis=1) * 100

	for idx, image in enumerate(test_set_x):
		img_mean = numpy.mean(image)	
		for idy, pixel in enumerate(image):
			test_set_x[idx][idy] = test_set_x[idx][idy] - img_mean
	
	test_set_x = normalize(test_set_x, axis=1) * 100
	
	# Find the mean and standard deviation of each pixel in the normalized training set
	# Subtract this mean from each pixel in all images
	# Divide each pixel value by its standard deviation

	train_set_x = numpy.transpose(train_set_x)
	valid_set_x = numpy.transpose(valid_set_x)
	test_set_x = numpy.transpose(test_set_x)

	for idx, x in enumerate(train_set_x):
		mean = numpy.mean(x)
		var = numpy.var(x)
		for idy, y in enumerate(x):
			train_set_x[idx][idy] = train_set_x[idx][idy] - mean
			train_set_x[idx][idy] = train_set_x[idx][idy] / var
	
	for idx, x in enumerate(valid_set_x):
		mean = numpy.mean(x)
		var = numpy.var(x)
		for idy, y in enumerate(x):
			valid_set_x[idx][idy] = valid_set_x[idx][idy] - mean
			valid_set_x[idx][idy] = valid_set_x[idx][idy] / var
	
	for idx, x in enumerate(test_set_x):
		mean = numpy.mean(x)
		var = numpy.var(x)
		for idy, y in enumerate(x):
			test_set_x[idx][idy] = test_set_x[idx][idy] - mean
			test_set_x[idx][idy] = test_set_x[idx][idy] / var
	
	# Transpose back before returning
	train_set_x = numpy.transpose(train_set_x)
	valid_set_x = numpy.transpose(valid_set_x)
	test_set_x = numpy.transpose(test_set_x)

	ret_val = [ [train_set_x, train_set_y], [valid_set_x, valid_set_y], [test_set_x, test_set_y] ]
	return ret_val

def load_data():
	train_set_x, train_set_y = load_set('/home/cduvedi/theano/train.csv')
	test_set_x, test_set_y = load_set('/home/cduvedi/theano/test.csv')
	valid_set_x, valid_set_y = load_set('/home/cduvedi/theano/valid.csv')
	
	print '... loaded data'
	print 'train: ', len(train_set_x)
	print 'test: ', len(test_set_x)
	print 'valid: ', len(valid_set_x)

	ret_val = [ [train_set_x, train_set_y], [valid_set_x, valid_set_y], [test_set_x, test_set_y] ]
	return ret_val

def write_data(datasets):
	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]
	
	train_file = open('train_preprocessed.csv', 'w')
	for idy, y in enumerate(train_set_y):
		train_file.write(str(y) + ',')
		for idx, x in enumerate(train_set_x[idy]):
			train_file.write(str(train_set_x[idy][idx]) + ' ')
		train_file.write('\n')
	train_file.close()

	valid_file = open('valid_preprocessed.csv', 'w')
	for idy, y in enumerate(valid_set_y):
		valid_file.write(str(y) + ',')
		for idx, x in enumerate(valid_set_x[idy]):
			valid_file.write(str(valid_set_x[idy][idx]) + ' ')
		valid_file.write('\n')
	valid_file.close()

	test_file = open('test_preprocessed.csv', 'w')
	for idy, y in enumerate(test_set_y):
		test_file.write(str(y) + ',')
		for idx, x in enumerate(test_set_x[idy]):
			test_file.write(str(test_set_x[idy][idx]) + ' ')
		test_file.write('\n')
	test_file.close()

if __name__ == '__main__':
	datasets = load_data()
	datasets = preProcessImages(datasets)	
	write_data(datasets)


