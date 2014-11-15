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
	print "Hello World"
#	labelledData = readLabelledDataFromCSV('/root/CS229-project/Datasets/train_small.csv')
#	labelledTestData = readLabelledDataFromCSV('/root/CS229-project/Datasets/test.csv')
#	pca = extractPCA( labelledData, labelledTestData )
#	pcaFeatureVectors = mapRawFeaturesToPCAFeatures( labelledData, pca )
#	writeFeatureVectorsToFile('train.feat', pcaFeatureVectors)
#	classifier = trainSVM(pcaFeatureVectors, labelledData.labels)
#	pcaTestVectors = mapRawFeaturesToPCAFeatures( labelledTestData, pca )
#	testSVM(pcaTestVectors, labelledTestData.labels, classifier)
	
#====================================================
# Helper functions. 
#====================================================

def readLabelledDataFromCSV(fileName):
	labelledData = LabelledData()

	fileHandle = open(fileName, 'r')
	reader = csv.reader(fileHandle)

	for row in reader:
		labelStr, featureStr, tp = row
		label = int(labelStr)
		features = map(lambda x: float(x), featureStr.split(' '))
		labelledData.addDataSample(label, features)

	# Debug purposes
	#for i in range(0, 7):
#		print len(labelledData.labelToFeatureVectors[i])

	return labelledData

def extractPCA(labelledData, testData):
	#randomFeatureVectors =  labelledData.getRandomFeatureVectors(numSamplesPerLabel = 10)
	randomFeatureVectors =  labelledData.featureVectors
	randomFaceArray = numpy.array(randomFeatureVectors)
	faceMean = numpy.mean(randomFaceArray, 0)
	facesAdjusted = randomFaceArray - faceMean
	for i in range(0, len(labelledData.featureVectors)):
		for j in range(0, len(labelledData.featureVectors[i])):
			labelledData.featureVectors[i][j] - labelledData.featureVectors[i][j] - faceMean[j]

	for i in range(0, len(testData.featureVectors)):
		for j in range(0, len(testData.featureVectors[i])):
			testData.featureVectors[i][j] = testData.featureVectors[i][j] - faceMean[j]

	pca = RandomizedPCA(n_components = 65, whiten=True).fit(facesAdjusted)
	print "PCA completed"
	print len(pca.components_)
	return pca

def mapRawFeaturesToPCAFeatures(labelledData, pca):
	return pca.transform(labelledData.featureVectors)

def readFeatureVectorsFromFile(fileName):
	pass

def writeFeatureVectorsToFile(fileName, featureVectors):
	fileHandle = open(fileName, 'w')
	for v in featureVectors:
		 fileHandle.write("%s\n" % v)

	fileHandle.close()

def trainSVM(featureVectors, labels):
	expr_classifier = svm.SVC(C = 0.0001)
	expr_classifier.fit(featureVectors, labels)
	print("Number of support vectors")
	print(expr_classifier.n_support_)
	return expr_classifier

def testSVM(testVectors, labels, classifier):
	numCorrect = 0
	index = 0
	fileHandle = open('labels.txt', 'w')

	for testVector in testVectors:
		predictedLabel = classifier.predict(testVector)
		fileHandle.write("%s \t %s \n" % (predictedLabel, labels[index]))
		if (predictedLabel == labels[index]):
			numCorrect = numCorrect + 1
		index = index + 1	
	print numCorrect
	fileHandle.close()


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

