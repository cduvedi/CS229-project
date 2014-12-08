#!/usr/bin/python

import csv
import numpy
import pylab as pl
import os
import random
import scipy
import sys

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
from sklearn.externals import joblib

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
	num_args = len(sys.argv)
	if (num_args > 5) :
		training_set_file = sys.argv[1]
		test_set_file = sys.argv[2]
		enable_training = bool(int(sys.argv[3]))
		enable_testing = bool(int(sys.argv[4]))
		compute_pca = bool(int(sys.argv[5]))
		svm_params_file = sys.argv[6] if (num_args > 6) else ""

		if ((not enable_training) and (svm_params_file == "")):
			print("[ERROR]: Since training has been disabled, you need to provide an SVM parameters file")
			sys.exit()

		print("Training set : " + training_set_file)
		print("Test set     : " + test_set_file)
		print("Enable training : " + str(enable_training))
		print("Enable testing  : " + str(enable_testing))
		print("Compute PCA     : " + str(compute_pca))
		print("SVM params file : " + svm_params_file)

		labelledTrainingData = readLabelledDataFromCSV(training_set_file)
		labelledTestData = readLabelledDataFromCSV(test_set_file)
		trainingFeatureVectors = labelledTrainingData.featureVectors
		testFeatureVectors = labelledTestData.featureVectors
		trainingSetFileName = os.path.basename(training_set_file).split(".")[0]

		if (compute_pca):
			print("Computing PCA and projecting training and test sets to PCA space")
			pca = extractPCA( labelledTrainingData, labelledTestData )
			trainingFeatureVectors = mapRawFeaturesToPCAFeatures( labelledTrainingData, pca )
			testFeatureVectors = mapRawFeaturesToPCAFeatures( labelledTestData, pca )
			writeFeatureVectorsToFile(trainingSetFileName + "_pca.csv", trainingFeatureVectors)
		else:
			print("Skipping PCA calculation")

		classifier = ()
		if (enable_training):
			print("Training SVM")
			classifier = trainSVM(trainingFeatureVectors, labelledTrainingData.labels)
			dumpSVMParametersToFile(classifier, trainingSetFileName + ".svm")
		else:
			print("Reading SVM parameters from file '" + svm_params_file + "'")
			classifier = readSVMParametersFromFile(svm_params_file)

		if (enable_testing):
			print("Running the trained SVM on the test set")
			testSVM(testFeatureVectors, labelledTestData.labels, classifier)
		else:
			print("Skipping testing")
	else:
		print("Invalid number of args")
		print("Run the script as follows:")
		print("         <script> <trainingSet> <testSet> <enable_training> <enable_testing> <compute_pca> <svm_params_file (Required if training is disabled)>")
	
#====================================================
# Helper functions. 
#====================================================

def readLabelledDataFromCSV(fileName):
	print("Reading labelled data from '" + fileName + "'")
	labelledData = LabelledData()

	fileHandle = open(fileName, 'r')
	reader = csv.reader(fileHandle)

	for row in reader:
		labelStr, featureStr, tp = row
		label = int(labelStr)
		features = map(lambda x: float(x), featureStr.split(' '))
		labelledData.addDataSample(label, features)

	# Debug purposes
	for i in range(0, 7):
		numSamples = len(labelledData.labelToFeatureVectors[i])
		print("# of samples for label " + str(i) + " = " + str(numSamples))

	return labelledData

def extractPCA(labelledData, testData):
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
	print("PCA completed")
	print("# of PCA components = " + str(len(pca.components_)))
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
	param_grid = {'C': [1e3, 5e3], 'gamma': [0.0001, 0.1], }
	#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	expr_classifier = GridSearchCV(svm.SVC(kernel='rbf', class_weight='auto'), param_grid)
#	expr_classifier = svm.SVC(C = 0.0001)
	expr_classifier.fit(featureVectors, labels)
#	print("Number of support vectors")
#	print(expr_classifier.n_support_)
	print("Best estimator found by grid search:")
	#print(clf.best_estimator_)
	print(expr_classifier.best_estimator_)
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
	print("# of correct labels in test set: " + str(numCorrect))
	fileHandle.close()

def readSVMParametersFromFile(fileName):
	return joblib.load(fileName)

def dumpSVMParametersToFile(classifier, fileName):
	joblib.dump(classifier, fileName)

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

