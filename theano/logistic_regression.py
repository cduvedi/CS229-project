#!/usr/bin/env python
#import theano
from theano import shared, sandbox, Out
import theano.tensor as T
import numpy
import time
from theano import function
from theano import shared
import csv
#f = function([], sandbox.cuda.basic_ops.gpu_from_host( T.exp(x)))
filehandle = open('/home/cduvedi/theano/train_small.csv', 'r')
rng = numpy.random
reader = csv.reader(filehandle)

X = list()
Y = list()

for row in reader:
	labelStr, featureStr = row
	label = int(labelStr)
	features = map(lambda x: float(x), featureStr.split(' '))
	X.append(features)
	Y.append(label)	

N = len(Y)
feats = len(X[0])

print N, feats
#D = ( rng.randn(N, feats), rng.randint(size=N, low = 0, high = 2) )
D = ( X, Y )
trainingSteps = 100

# Declare symbolic representations for
x = T.matrix('x')				# Input vectors
y = T.vector('y')				# Labels
w = shared(rng.randn(feats), name='w')	# The weight vector for logistic regression initialized to random
b = shared(0., name='b')			# THe intercept term initialized to 0

# Construct the expression graph
p = 1 / (1 + T.exp(-T.dot(x, w) - b) )		# The logistic function
prediction = p > 0.5				# PRedict based on the value 
x_entropy = -y * T.log(p) - (1-y) * T.log(1 - p)# THe cross entropy
cost = x_entropy.mean() + 0.01 * (w ** 2).sum()	# Cost minimization function
gw, gb = T.grad(cost, [w, b])			# The gradient, used to update w and b

#Compile the train and predict functions
train = function(
	inputs = [x, y],
	outputs=[Out(sandbox.cuda.basic_ops.gpu_from_host(T.cast(prediction, 'float32')),borrow=True), Out(sandbox.cuda.basic_ops.gpu_from_host(T.cast(x_entropy, 'float32')), borrow=True)],
#	outputs = [prediction, x_entropy],
	updates = ( (w, w - 0.1 * gw), (b, b - 0.1 * gb) ),
	allow_input_downcast=True)
predict = function(inputs = [x], outputs = prediction, allow_input_downcast=True)	

#Train
for ittr in range(0, trainingSteps):
	print ittr
	pred, err = train(D[0], D[1])

num_correct = 0
predicted_y = predict(D[0])
for ittr in range(0, N):
	if D[1][ittr] == predicted_y[ittr]:
		num_correct = num_correct + 1

print num_correct
