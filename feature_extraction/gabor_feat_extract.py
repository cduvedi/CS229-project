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

from skimage import data
from skimage.util import img_as_float
from skimage.filter import gabor_kernel

import eigen_faces_refactored

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
#	    print k, "Kernal_len ", len(kernel), "Image_len ", len(image.rank) 
	    filtered = nd.convolve(image, kernel, mode='wrap')
	    feats[k, 0] = filtered.mean()
	    feats[k, 1] = filtered.var()
    return feats


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i


# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

labelledTrainData = eigen_faces_refactored.readLabelledDataFromCSV('../Datasets/train_small.csv')

trndata= []
for i in range(0,len(labelledTrainData.featureVectors)):
	trndata.append(labelledTrainData.featureVectors[i])

print len(trndata), " ", len(trndata[0])
trn_2D = np.reshape(np.array(trndata[0]), (48, 48))

shrink = (slice(0, None, 3), slice(0, None, 3))
brick_1 = img_as_float(data.load('C:\Users\Auro\Documents\Acads\Autumn 14\CS229\Project\CS229-project\/feature_extraction\cg.jpg'))[shrink]

brick_11 = {}

for i in range(0,len(brick_1)):
	for j in range(0,len(brick_1[0])):
		brick_11[i,j] = brick_1[i,j][1]*255

brick = np.array(brick_11)

image_names = ('brick')
images = (brick)

# prepare reference features
ref_feats = np.zeros((3, len(kernels), 2), dtype=np.double)
ref_feats[0, :, :] = compute_feats(trn_2D, kernels)

print('Rotated images matched against references using Gabor filter banks:')

print('original: brick, rotated: 30deg, match result: ') #, end='')
feats = compute_feats(nd.rotate(trn_2D, 45, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: brick, rotated: 70deg, match result: ') #, end='')
feats = compute_feats(nd.rotate(trn_2D, angle=70, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])



