import os
import csv
import random
import Image
import math
import pylab as pl
from sklearn.decomposition import RandomizedPCA
#import eigen_faces_refactored

import matplotlib
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as nd
from scipy.misc import toimage

from skimage import data
from skimage.util import img_as_float
from skimage.filter import gabor_kernel

#import eigen_faces_refactored as eig

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    filtered_list = list();
    for k, kernel in enumerate(kernels):
	    filtered = nd.convolve(image, kernel, mode='wrap')
	    filtered_list.append(filtered)
    return filtered_list


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i


def gabor():
	# prepare filter bank kernels
	kernels = []
	sigma = 6.0
	for f in range(9):
	    frequency = 1 / (2 * pow(math.sqrt(2), f) )
	    for theta in range(0, 8):
	        theta = (np.pi * theta) / 8.0
	        kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
	        kernels.append(kernel)

	fileHandle = open('train_small.csv', 'r')
	reader = csv.reader(fileHandle)

	#gabor_features = list()
	grid_img = Image.new('P', (432, 384))

	gab_file = open('gabor_feats.csv', 'w')
	for row in reader:
		labelStr, featureStr, tp = row
		label = int(labelStr)
		features = map(lambda x: float(x), featureStr.split(' '))
		
		trn_2D = np.reshape(np.array(features), (48, 48))
		
		gab_images =  compute_feats(trn_2D, kernels)
		#with open("gabor_feats.csv", "a") as f:
		gab_file.write(labelStr + ',')
		pixels_str = ""
		for img in gab_images:
			img_array = img.ravel()
			for pixel in img_array:
				pixels_str = pixels_str + (" %.4f" % pixel)
				#gab_file.write(s + ' ')
		gab_file.write(pixels_str.lstrip() + '\n')
		#gab_file.write('\n')
	gab_file.close()	
	#for i in range(0, 9):
#		for j in range(0, 8):
#			grid_img.paste(toimage(gab_images[i*8 + j]), (i * 48, j * 48))
#	grid_img.show()
	fileHandle.close()
	
	print "Finished extracting gabor features"

gabor()
