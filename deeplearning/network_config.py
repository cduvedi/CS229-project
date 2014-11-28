# network_config.py - Create network symbolic graph and train it

# Common imports
import os
import sys
import time
import numpy

# Theano imports
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

# Imorts from other custom modules
from mlp import HiddenLayer # The hidden layer is needed as the second last layer of the deep network
import learning_utils
import utils
from utils import *

def configureAndTrainDeepNN(learningRate, n_epochs, batch_size):
	"""
	Main top level function for the neural network
	Arguments
	1. learningRate - for the neural network, generally 0.1
	2. n_epoch - number of epochs for training
	3. utils.nkerns - tupple having the number of kernels for each convolutional layer
	4. batch_size - top optimize performance
	"""

	randGen = numpy.random.RandomState(23455)

	# Load datasets -  assume that image preprocessing is already done
	datasets = learning_utils.load_data()

	# split into training, testing, validation tests
	trainSetX, trainSetY = datasets[0]
	validSetX, validSetY = datasets[1]
	testSetX, testSetY = datasets[2]

	# Calculate the number of batches
	n_train_batches = trainSetX.get_value(borrow=True).shape[0]
	n_valid_batches = validSetX.get_value(borrow=True).shape[0]
	n_test_batches = testSetX.get_value(borrow=True).shape[0]

	n_train_batches /= batch_size
	n_test_batches /= batch_size
	n_valid_batches /= batch_size

	print n_train_batches
	index = T.lscalar()	# batch index

	# Declare input(matrix) and output(1D vector of labels)
	x = T.matrix('x')
	y = T.ivector('y')

	##################################################
	# Definte the actual network
	##################################################

	# layer declarations

	# Layer0
	layer0Input = x.reshape( ( batch_size, 1, layer0InSize[0], layer0InSize[1] ) )

	layer0 = utils.convLayer(
		randGen,
		input = layer0Input,
		imageShape = ( batch_size, 1, layer0InSize[0], layer0InSize[1] ),
		filterShape = ( utils.nkerns[0], 1, kern0Size[0], kern0Size[1] ),
		poolSize = poolSize0
	)

	# Layer 1
	layer1Input = layer0.output

	layer1 = utils.convLayer(
		randGen, 
		input = layer1Input,
		imageShape = ( batch_size, utils.nkerns[0], layer1InSize[0], layer1InSize[1] ),
		filterShape = ( utils.nkerns[1], utils.nkerns[0], kern1Size[0], kern1Size[1] ),
		poolSize = poolSize1
	)

	# Layer 2
	layer2Input = layer1.output

	layer2 = utils.convLayer(
		randGen, 
		input = layer2Input,
		imageShape = ( batch_size, utils.nkerns[1], layer2InSize[0], layer2InSize[1] ),
		filterShape = ( utils.nkerns[2], utils.nkerns[1], kern2Size[0], kern2Size[1] ),
		poolSize = poolSize2
	)

	# Layer 3 - Fully connected layer
	layer3Input = layer2.output.flatten(2)

	layer3 = HiddenLayer(
		randGen,
		input = layer3Input,
		n_in = layer3InSize,
		n_out = numHiddenLayerNeurons,
		activation = T.tanh
	)

	# Layer 4 -Output layer
	layer4Input = layer3.output

	layer4 = learning_utils.LogisticRegression(
		input = layer4Input,
		n_in = numHiddenLayerNeurons,
		n_out = numClasses
	)

	##################################################
	# Now that the network is defined, define the update rules
	##################################################
	
	# Cost ot minimize
	cost = layer4.negative_log_likelihood(y)

	# Testing and validation of the model
	# This will not be done in every iteration
	testModel = theano.function(
		[index],
		layer4.errors(y),
		givens = {
			x: testSetX[index * batch_size: (index + 1) * batch_size ],
			y: testSetY[ index * batch_size: (index + 1) * batch_size ]
		}
	)

	validateModel = theano.function(
		[index],
		layer4.errors(y),
		givens = {
			x: validSetX[ index * batch_size: (index + 1) * batch_size ],
			y: validSetY[ index * batch_size: (index + 1) * batch_size ]
		}
	)

	params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

	# Calculate the gradients for stochastic gradient descent
	grads = T.grad(cost, params)

	# Define the updates to the parameters according to gradient descent
	updates = [
		(param_i, param_i - learningRate * grad_i)
		for param_i, grad_i in zip(params, grads)
	]

	# Train the model
	trainModel = theano.function(
		[index],
		cost,
		updates = updates,
		givens = {
			x: trainSetX[ index * batch_size: (index + 1) * batch_size ],
			y: trainSetY[ index * batch_size: (index + 1) * batch_size ]
		}
	)


	## Done building the symbolic graph
	# Actual code to train the network

	# First define some parameters for early termination
	patience = 10000	# Minimum number of samples to scan
	patience_increase = 2	# Minimum wait when a new best is found

	improvement_threshold = 0.995	# The minimum reduction in error that will be considered significant

	validation_frequency = min(n_train_batches, patience / 2)	# Validating at every minibatch will be wasted effort

	best_validation_loss = numpy.inf	# Start with a loss of infinity
	best_iter = 0				# At the 0th iteration
	test_error_rate = 0.
	start_time = time.clock()
	print "... Training the network", start_time

	epoch = 0
	done_looping = False			# Used to determine the early termination conditions


	while(epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1

		for minibatch_index in xrange(n_train_batches):	# Loop over all minibatches - will improve GPU performance
			iter = (epoch - 1) * n_train_batches + minibatch_index # The absolute iteration counter
			print "Training iteration# ", iter

			cost_ij = trainModel(minibatch_index)	# Train for this minibatch

			if(iter + 1) % validation_frequency == 0:	# Is validation required at this iteraion?

				validation_losses = [validateModel(i) for i in xrange(n_valid_batches) ]

				validation_loss_mean = numpy.mean(validation_losses)
				print "epoch ", epoch, " minibatch ", minibatch_index+1, "/", n_train_batches, " validation error ", validation_loss_mean * 100

				# If this is the new best validation score
				if validation_loss_mean < best_validation_loss:
					# Increase the patience if the best validation score has been found late and the improvement is good enough

					if validation_loss_mean < best_validation_loss * improvement_threshold:
						patience = max(patience, iter * patience_increase)

					# Save the new best validation score and the iteration at which this was obtained
					best_validation_loss = validation_loss_mean
					best_iter = iter

					# Display the test error also
					test_losses = [testModel(i) for i in xrange(n_test_batches)]
					test_error_rate = numpy.mean(test_losses)
					print "epoch ", epoch, " minibatch ", minibatch_index+1, "/", n_train_batches, " test error ", test_error_rate * 100
			if patience <= iter:
				done_looping = True
				break

	end_time = time.clock()
	print "Optimization complete at ", end_time

	print "Best validation error ", best_validation_loss
	print "Test error for best validation model ", test_error_rate
	
