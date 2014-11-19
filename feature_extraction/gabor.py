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
                #print "frequency", frequency, "theta", theta
	    #for sigma in range(0, 8):
            #    sigma = 1.0 + (sigma * 0.625)
		# TODO: check effect of frequency on kernel size
	        #for frequency in (0.1, 0.2):
	        kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
	        kernels.append(kernel)

	fileHandle = open('../Datasets/train_head.csv', 'r')
	reader = csv.reader(fileHandle)

	#gabor_features = list()
	grid_img = Image.new('P', (432, 384))

	for row in reader:
		labelStr, featureStr = row
		label = int(labelStr)
		features = map(lambda x: float(x), featureStr.split(' '))
		
		trn_2D = np.reshape(np.array(features), (48, 48))
		
		gab_images =  compute_feats(trn_2D, kernels)
		print "len_gab_img", len(gab_images)
		print "len_kernels", len(kernels)
		#print gab_images[0]
		#toimage(gab_images[1]).show()
		#toimage(trn_2D).show()

		break
		#avg_gab_image = np.zeros_like(gab_images[0])
		
		#for ittrImg in range(0, len(gab_images)):
		#	avg_gab_image = avg_gab_image + gab_images[ittrImg]		
		#
		#avg_gab_image = avg_gab_image / len(gab_images)
		#avg_gab_array = avg_gab_image.ravel()

		#gabor_features.append(avg_gab_array)
		#print len(gabor_features)

	#	with open("gabor_feats.csv", "a") as f:
	#		writer = csv.writer(f)
	#		writer.writerow(avg_gab_array)
	for i in range(0, 9):
		for j in range(0, 8):
			grid_img.paste(toimage(gab_images[i*8 + j]), (i * 48, j * 48))
	grid_img.show()
	fileHandle.close()
	
	print "Finished extracting gabor features"

#	print "Going for dimensionality reduction using PCA"
#	pca = RandomizedPCA(n_components = 65).fit(np.array(gabor_features))
	#toimage(avg_gab_image).show()
	#toimage(gab_images[0]).show()
	
#	pcaFeatureVectors = eig.mapRawFeaturesToPCAFeatures( labelledTrainData, pca )

#	return(pca, pcaFeatureVectors)
#eig.writeFeatureVectorsToFile('train.feat', pcaFeatureVectors)
#classifier = eig.trainSVM(pcaFeatureVectors, labelledTrainData.labels)
#pcaTestVectors = mapRawFeaturesToPCAFeatures( labelledTestData, pca )
#testSVM(pcaTestVectors, labelledTestData.labels, classifier)

gabor()
