import csv
import os
from sklearn.externals.six.moves import zip
import matplotlib
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import toimage
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import eigen_faces_refactored_adaboost as eigen_faces_refactored

def extract_ada(labelledTrainData): #, labelledValData):
	print "Create Adaboost calssifier"
	bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1), n_estimators = 500, learning_rate = 1, algorithm = "SAMME.R")
	print "Fit the Data and the labels using training data"
	print len(labelledTrainData.featureVectors), ":", len(labelledTrainData.featureVectors[0]),":",len(labelledTrainData.labels) 
	bdt_discrete.fit(labelledTrainData.featureVectors, labelledTrainData.labels)

	print "Fitting done"

	#discrete_test_errors = []
	#for discrete_train_predict in bdt_discrete.staged_predict(labelledValData.featureVectors):
	#	print "Discrete Train Predict: ", discrete_train_predict
	#	discrete_test_errors.append(1. - accuracy_score(discrete_train_predict, (labelledValData.labels)))
	#	print len(discrete_test_errors)

	n_discrete_trees = len(bdt_discrete)
	print n_discrete_trees

	# Boosting might terminate early, but the following arrays are always
	# n_estimators long. We crop them to the actual number of trees here:
	#discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_discrete_trees]
	#discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_discrete_trees]
	#print "Estimator error:", len(discrete_estimator_errors)
	#print "Estimator weights:", len(discrete_estimator_weights)


	num_features = len(bdt_discrete.feature_importances_)
	#top_feature_indices = sorted(range(num_features)), key=lambda k: -1 * bdt_discrete.feature_importances_[k])
	#sorted_feature_imp = sorted(bdt_discrete.feature_importances_, key = lambda x: -1 * x)
	print "Sorting the features based on their importance values.."
	sorted_feature_imp =  sorted(zip(range(num_features), bdt_discrete.feature_importances_), key=lambda x : -1 * x[1])
	print "Writing sorted feature importances to the file..."
	##############################
	# MODIFY OUTPUT FILE NAME
	#############################
	o_file = "/farmshare/user_data/jithinpt/adaboost_feats_MaxD_1_NEst_5000_trainingSet.csv"
	print o_file
	with open(o_file,'a') as fh:
		writer = csv.writer(fh)
		for i in range(num_features):
			writer.writerow(sorted_feature_imp[i])
	#feature_imp = np.zeros((1,2304))
	#one_one = np.ones((1,2304))

	#for i in range(72):
	#	feature_imp = np.add(feature_imp, bdt_discrete.feature_importances_[i*2304:(i+1)*2304])

	#feature_imp = bdt_discrete.feature_importances_

	#feature_imp_2d = np.reshape((feature_imp[0]*255),(48,48))
	#feature_imp_2d = np.reshape((feature_imp*255),(48,48))
	#toimage(feature_imp_2d).show()
	

print "Reading Train Dataset 1"
################################
# MODIFY THE FILENAME BELOW
###################################
labelledTrainData = eigen_faces_refactored.readLabelledDataFromCSV('/farmshare/user_data/jithinpt/gabor_feats_cduvedi_lp.csv', 2)
#labelledTrainData = eigen_faces_refactored.readLabelledDataFromCSV('/farmshare/user_data/jithinpt/trainingSet.csv', 2)
#labelledTrainData = eigen_faces_refactored.readLabelledDataFromCSV('/farmshare/user_data/jithinpt/gabor_feats_4750', 0)
print "Reading Train Dataset 1 Done....."
print "Starting the ADABOOST feature selection..."
extract_ada(labelledTrainData) #, labelledTestData)
#
#print "Reading Train Dataset 2"
#labelledTrainData = eigen_faces_refactored.readLabelledDataFromCSV('/farmshare/user_data/jithinpt/gabor_feats_lp_9501_19000.csv')
#print "Reading Train Dataset 2 Done....."
#print "Starting the ADABOOST feature selection..."
#extract_ada(labelledTrainData) #, labelledTestData)
#
#print "Reading Train Dataset 3"
#labelledTrainData = eigen_faces_refactored.readLabelledDataFromCSV('/farmshare/user_data/jithinpt/gabor_feats_19001_28709_lp.csv')
#print "Reading Train Dataset 3 Done....."
#print "Starting the ADABOOST feature selection..."
#extract_ada(labelledTrainData) #, labelledTestData)
