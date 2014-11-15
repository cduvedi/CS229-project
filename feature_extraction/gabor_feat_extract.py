import os
import csv
import random

import pylab as pl
from sklearn.decomposition import RandomizedPCA
import eigen_faces_refactored

import matplotlib
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as nd
from scipy.misc import toimage

from skimage import data
from skimage.util import img_as_float
from skimage.filter import gabor_kernel

import eigen_faces_refactored as eig

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


def gabor(labelledTrainData):
	# prepare filter bank kernels
	kernels = []
	for theta in range(4):
	    theta = theta / 4. * np.pi
	    for sigma in (5, 10):
		# TODO: check effect of frequency on kernel size
	        #for frequency in (0.05, 0.25):
	        for frequency in (0.1, 0.2):
	            kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
	            kernels.append(kernel)

	gabor_features = list()
	for ittr in range(0, len(labelledTrainData.featureVectors)):
		trn_2D = np.reshape(np.array(labelledTrainData.featureVectors[ittr]), (48, 48))
		
		# prepare reference features
		ref_feats = np.zeros((3, len(kernels), 2), dtype=np.double)
		
		gab_images =  compute_feats(trn_2D, kernels)
		avg_gab_image = np.zeros_like(gab_images[0])
		for ittrImg in range(0, len(gab_images)):
			avg_gab_image = avg_gab_image + gab_images[ittrImg]		
		
		avg_gab_image = avg_gab_image / len(gab_images)
		
		avg_gab_array = avg_gab_image.ravel()
		gabor_features.append(avg_gab_array)

	print "Finished extracting gabor features"

	with open("gabor_feats.csv", "wb") as f:
		writer = csv.writer(f)
		writer.writerows(gabor_features)
	
	print "Going for dimensionality reduction using PCA"
	pca = RandomizedPCA(n_components = 65).fit(np.array(gabor_features))
	#toimage(avg_gab_image).show()
	#toimage(gab_images[0]).show()
	
	pcaFeatureVectors = eig.mapRawFeaturesToPCAFeatures( labelledTrainData, pca )

	return(pca, pcaFeatureVectors)
#eig.writeFeatureVectorsToFile('train.feat', pcaFeatureVectors)
#classifier = eig.trainSVM(pcaFeatureVectors, labelledTrainData.labels)
#pcaTestVectors = mapRawFeaturesToPCAFeatures( labelledTestData, pca )
#testSVM(pcaTestVectors, labelledTestData.labels, classifier)
