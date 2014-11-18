from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

import os
import csv
import numpy
import random
import pylab as pl
from sklearn.decomposition import RandomizedPCA
import eigen_faces_refactored
import gabor_feat_extract as gab

###################################
# Use features from PCA extraction
###################################
#labelledTrainData = eigen_faces_refactored.readLabelledDataFromCSV('../Datasets/train_med.csv')
#pcaTrain = eigen_faces_refactored.mapRawFeaturesToPCAFeatures( labelledTrainData )
#pcaTrainFeatureVectors = pcaTrain.transform(labelledTrainData.featureVectors)

#labelledTestData = eigen_faces_refactored.readLabelledDataFromCSV('../Datasets/test.csv')
#pcaTestFeatureVectors = pcaTrain.transform(labelledTestData.featureVectors)

###################################
# Use features from Gabor filter
###################################

print "Gabor Feature extraction"
labelledTrainData = eigen_faces_refactored.readLabelledDataFromCSV('../Datasets/train.csv')
pcaTrain, pcaTrainFeatureVectors = gab.gabor(labelledTrainData)

labelledTestData = eigen_faces_refactored.readLabelledDataFromCSV('../Datasets/test.csv')
pcaTestFeatureVectors = pcaTrain.transform(labelledTestData.featureVectors)


###########################################################################
#	Neural Network
###########################################################################
trnDS = ClassificationDataSet(len(pcaTrainFeatureVectors[0]), nb_classes=7)
tstDS = ClassificationDataSet(len(pcaTestFeatureVectors[0]), nb_classes=7)

for i in range(0,len(pcaTrainFeatureVectors)):
	trnDS.appendLinked(pcaTrainFeatureVectors[i], labelledTrainData.labels[i])

for i in range(0,len(pcaTestFeatureVectors)):
	tstDS.appendLinked(pcaTestFeatureVectors[i], labelledTestData.labels[i])

trnDS._convertToOneOfMany(bounds=[0, 1])
tstDS._convertToOneOfMany(bounds=[0, 1])

trndata = trnDS
tstdata = tstDS

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim

fnn = buildNetwork( trndata.indim, 2, trndata.outdim, outclass=SoftmaxLayer , bias = 'True')

trainer = BackpropTrainer( fnn, dataset=trndata, learningrate = 0.001, momentum=0.99, verbose=True, weightdecay=0.01)

####################################################
#	Training and Testing
####################################################
trainer.trainUntilConvergence(verbose=True, trainingData = trndata, validationData = tstdata, maxEpochs=50)

# stop training and start using my trained network here

def maxConfIndex(arr):
	maxIndex = 0
	for i in range(len(arr)):
		if (arr[i] > arr[maxIndex]):
			maxIndex = i
	return maxIndex

j= 0
k = 0

for i in range(0,len(pcaTestFeatureVectors)):
	output = fnn.activate(pcaTestFeatureVectors[i])
	print "Result - ", output
	if labelledTestData.labels[i] != maxConfIndex(output):
		k = k + 1
	else:
		j = j + 1
		
print "Correct predictions : ", j
print "Incorrect predictions : ", k
