#!/usr/bin/env python
from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time
from theano import function
from theano import shared
#f = function([], sandbox.cuda.basic_ops.gpu_from_host( T.exp(x)))

x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = function( [x], s)

out = logistic( [ [0, 1], [-1, -2] ] )
print out

a, b, c = T.dmatrices('a', 'b', 'c')
diff = a - b - c
abs_diff = abs(diff)
diff_sq = diff ** 2
f1 = function( [a, b, c], [diff, abs_diff, diff_sq])

out = f1( [ [1, 1], [1, 1] ], [ [2, 2], [2, 2] ], [ [3, 3], [3, 3] ])
print out

state = shared( numpy.zeros(2) )
inc = T.ivector()
accumulator = function( [inc], state, updates=[(state, state + inc)])

val = state.get_value()
print val

accumulator( [1, 1] )
val = state.get_value()
print val

accumulator( [2, 2] )
val = state.get_value()
print val
