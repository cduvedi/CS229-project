from sklearn.externals.six.moves import zip

import matplotlib.pyplot as plt

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import eigen_faces_refactored 

def extract_ada(labelledTrainData, labelledValData):

	bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2), n_estimators = len(labelledValData.featureVectors[0]), learning_rate = 1.5, algorithm = "SAMME")
	bdt_discrete.fit(labelledTrainData.featureVectors, labelledTrainData.labels)

	discrete_test_errors = []
	for discrete_train_predict in zip(bdt_discrete.staged_predict(labelledValData.featureVectors)):
		discrete_test_errors.append(1. - accuracy_score(discrete_train_predict[0], (labelledValData.labels)))

	n_discrete_trees = len(bdt_discrete)
	print n_discrete_trees

	# Boosting might terminate early, but the following arrays are always
	# n_estimators long. We crop them to the actual number of trees here:
	discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_discrete_trees]
	discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_discrete_trees]
	print "Estimator error:", len(discrete_estimator_errors)
	print "Estimator weights:", len(discrete_estimator_weights)


#fh_train = open('gabor_feats.csv', 'r')
#fh_val = open('gabor_feats_test_small.csv', 'f')
labelledTrainData = eigen_faces_refactored.readLabelledDataFromCSV('file-0')
labelledTestData = eigen_faces_refactored.readLabelledDataFromCSV('gabor_feats_test_smaller.csv')

extract_ada(labelledTrainData, labelledTestData)
