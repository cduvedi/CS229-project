#!/usr/bin/env python
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
import utils
import network_config

if __name__ == '__main__':
	network_config.configureAndTrainDeepNN(learningRate=0.1, n_epochs=200, batch_size=500)
