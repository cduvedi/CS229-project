# utils.py - common functions / classes required for deep learning examples

# Common imports
import os
import sys
import time
import numpy
import sklearn
from sklearn.preprocessing import normalize

# Theano imports
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

# Imorts from other custom modules
from mlp import HiddenLayer # The hidden layer is needed as the second last layer of the deep network

#Class definitions for different components of the deep network
"""
The network needs 4 types of layers
1. Convolution layer
2. Convolution layers with max-pooling / reduction
3. Fully connected layer
4. Output layer (for now sigmoidal)

For now the layer without maxpooling is implemented using a poolsize of (1, 1)
"""
## TODO - create a common class called "Layer" and make the following inherit from it

nkerns = [32, 32, 64]

# Class for convolition layer
class convLayer(object):
	"""
	Initialization for the class
	Arguments
	1. randGen - numpy.random.RandomState - used to generate initial weights
	2. input - 4D tensor of shape imageShape
	3. filterShape - tupple4, shape of the convolutional filters to be applied
			<number of filters, number of input maps, filter height, filter width>
	4. imageShape - tupple4
			<batch size, number of input maps, image height, imag width>
	5. poolSize - tupple2, used for reduction
	"""
	def __init__(self, randGen, input, filterShape, imageShape, poolSize):

		assert imageShape[1] == filterShape[1]
		self.input = input

		fan_in = numpy.prod(filterShape[1:])
		fan_out = numpy.prod(filterShape[0] * numpy.prod(filterShape[2:]) / numpy.prod(poolSize) )
		
		# Have to calculate w^T * x + b
		#Random weights initialization
		wBound = numpy.sqrt(6. / (fan_in + fan_out)) # HACK: FIXME - where did the 6 come from??
		# The weight has to be a shared variable because it will keep geting updates
		self.W = theano.shared (
			numpy.asarray(
				randGen.uniform(low = -wBound, high = wBound, size=filterShape),
				dtype = theano.config.floatX
			),
			borrow=True
		)

		# Initialize the bias(b) to zeros - one per output map
		bVal = numpy.zeros((filterShape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value = bVal, borrow=True)

	
		convOut = conv.conv2d(
			input = input,
			filters = self.W,
			filter_shape = filterShape,
			image_shape = imageShape
		)

		# Max pooling
		poolOut = downsample.max_pool_2d(
			input = convOut,
			ds = poolSize,
			ignore_border = True
		)
		# The non linear step
		self.output = T.tanh(poolOut + self.b.dimshuffle('x', 0, 'x', 'x'))

		self.params = [self.W, self.b]
# end of class convLayer

# Layer configurations

# Layer 0
layer0InSize = [48, 48]
kern0Size = [5, 5]
poolSize0 = (2, 2)

# Layer 1
layer1InSize = [(layer0InSize[0] - kern0Size[0] + 1) / poolSize0[0], (layer0InSize[1] - kern0Size[1] + 1) / poolSize0[1] ]
kern1Size = [4, 4]
poolSize1 = (1, 1)

# Layer 2
layer2InSize = [(layer1InSize[0] - kern1Size[0] + 1) / poolSize1[0], (layer1InSize[1] - kern1Size[1] + 1) / poolSize1[1] ]
kern2Size = [5, 5]
poolSize2 = (1, 1)

# Layer 3
layer3InSize = ( (layer2InSize[0] - kern2Size[0] + 1) / poolSize2[0]) * ((layer2InSize[1] - kern2Size[1] + 1) / poolSize2[1] ) * nkerns[2]
numHiddenLayerNeurons = 3*1024

# Layer 4
numClasses = 7


