#!/usr/bin/env python
from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time
from theano import function
#f = function([], sandbox.cuda.basic_ops.gpu_from_host( T.exp(x)))

print "Scalar addition"
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f1 = function( [x, y], z )

a = f1(2, 3)
print a

print "Matrix addition"
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f2 = function( [x, y], z )

b = f2( [ [1, 2], [3, 4] ], [ [10, 20], [30, 40] ] )
print b

print "Expression evaluation"
x = T.vector()
y = T.vector()
out = x ** 2 + y ** 2 + 2 * x * y
f3 = function( [x, y], out)

c = f3( [1, 2], [4, 5])
print c
