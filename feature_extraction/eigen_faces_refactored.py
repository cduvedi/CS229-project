import os
import csv
import numpy
import scipy
import random
import pylab as pl

from scipy import linalg
from scipy.misc import toimage
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn import svm
from sklearn.svm import SVC

#====================================================
# Terminology used:
# 	Sample/DataSample -> (Y,[Xs])
#   FeatureVector -> [Xs]
#   Label -> Y
#====================================================

#====================================================
# main() function in the script
#====================================================

def main():
	labelledData = readLabelledDataFromCSV('/Users/jithinpt/Documents/Acads/CS_229/Project/Datasets/Kaggle/fer2013/train_small.csv')
	pcaFeatureVectors = mapRawFeaturesToPCAFeatures( labelledData )
	writeFeatureVectorsToFile('train.feat', pcaFeatureVectors)
	trainSVM(pcaFeatureVectors, labelledData.labels)

#====================================================
# Helper functions. 
#====================================================

def readLabelledDataFromCSV(fileName):
	labelledData = LabelledData()

	fileHandle = open(fileName, 'r')
	reader = csv.reader(fileHandle)

	for row in reader:
		labelStr, featureStr = row
		label = int(labelStr)
		features = map(lambda x: float(x), featureStr.split(' '))
		labelledData.addDataSample(label, features)

	# Debug purposes
	for i in range(0, 7):
		print len(labelledData.labelToFeatureVectors[i])

	return labelledData

def mapRawFeaturesToPCAFeatures(labelledData):
	randomFeatureVectors =  labelledData.getRandomFeatureVectors(numSamplesPerLabel = 10)
	randomFaceArray = numpy.array(randomFeatureVectors)
	faceMean = numpy.mean(randomFaceArray, 0)
	facesAdjusted = randomFaceArray - faceMean

	pca = RandomizedPCA(n_components = 20, whiten=True).fit(facesAdjusted)
	print "PCA completed"
	print len(pca.components_)

	return pca.transform(labelledData.featureVectors)

def readFeatureVectorsFromFile(fileName):
	pass

def writeFeatureVectorsToFile(fileName, featureVectors):
	fileHandle = open(fileName, 'w')
	for v in featureVectors:
		 fileHandle.write("%s\n" % v)

	fileHandle.close()

def trainSVM(featureVectors, labels):
	expr_classifier = svm.SVC()
	expr_classifier.fit(featureVectors, labels)
	print("Number of support vectors")
	print(expr_classifier.n_support_)


#====================================================
# LabelledData class
#   Stores a set of labelled data points
#====================================================

class LabelledData():

	def __init__(self):
		self.labels = []
		self.featureVectors = []
		self.labelToFeatureVectors = {} # {Y1: [X1,X2,..], Y2: [X5,X6,..]}
		for i in range(7):
			self.labelToFeatureVectors[i] = []

	def getSample(self, index):
		if (index < len(labels)):
			return (self.labels[index], self.featureVectors[index])

		print("[WARNING]: Index (" + index + ") greater than number of samples (" + len(labels) + ") in the dataset")
		return ("", [])

	def addDataSample(self, label, features):
		self.labels.append(label)
		self.featureVectors.append(features)
		self.labelToFeatureVectors[label].append(features)

	def getRandomFeatureVectors(self, numSamplesPerLabel):
		randomFeatureVectors = []
		for label in self.labelToFeatureVectors:
			featureVectors = self.labelToFeatureVectors[label]
			numSamples = len(featureVectors)
			if (numSamples > numSamplesPerLabel):
				randomIndices = random.sample(range(numSamples), 10)
				for i in randomIndices:
					randomFeatureVectors.append(featureVectors[i])

		return randomFeatureVectors

	def size(self):
		return len(labels)

main()

