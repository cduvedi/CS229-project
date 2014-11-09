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

labelledTrainData = eigen_faces_refactored.readLabelledDataFromCSV('../Datasets/train_small.csv')
pcaTrain = eigen_faces_refactored.mapRawFeaturesToPCAFeatures( labelledTrainData )
pcaTrainFeatureVectors = pcaTrain.transform(labelledTrainData.featureVectors)

labelledTestData = eigen_faces_refactored.readLabelledDataFromCSV('../Datasets/test_small.csv')
#pcaTestFeatureVectors = eigen_faces_refactored.mapRawFeaturesToPCAFeatures( labelledTestData )
pcaTestFeatureVectors = pcaTrain.transform(labelledTestData.featureVectors)


trnDS = ClassificationDataSet(len(pcaTrainFeatureVectors[0]), nb_classes=7)
tstDS = ClassificationDataSet(len(pcaTestFeatureVectors[0]), nb_classes=7)

for i in range(0,len(pcaTrainFeatureVectors)):
	trnDS.appendLinked(pcaTrainFeatureVectors[i], labelledTrainData.labels[i])

for i in range(0,len(pcaTestFeatureVectors)):
	tstDS.appendLinked(pcaTestFeatureVectors[i], labelledTestData.labels[i])

trnDS._convertToOneOfMany(bounds=[0, 1])
tstDS._convertToOneOfMany(bounds=[0, 1])

#print DS.calculateStatistics()

gen, trndata = trnDS.splitWithProportion( 0 )
tstdata, gen = tstDS.splitWithProportion( 1 )

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
#print "First sample (input, target, class):"
#print trndata['input'][0], trndata['target'][0], trndata['class'][0]

fnn = buildNetwork( trndata.indim, 1, trndata.outdim, outclass=SoftmaxLayer , bias = 'True')

trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

trainer.trainUntilConvergence(maxEpochs=100)
# stop training and start using my trained network here

for i in range(0,len(pcaTestFeatureVectors)):
	output = fnn.activate(pcaTestFeatureVectors[i])
	print "Result - ", output


#for i in range(1,20):
#	trainer.trainEpochs( 1 )
#
#	trnresult = percentError( trainer.testOnClassData(), trndata['class'] )
#	tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )
#
#	print "epoch: %4d" % trainer.totalepochs, \
#			"  train error: %5.2f%%" % trnresult, \
#			"  test error: %5.2f%%" % tstresult
