#!/bin/env python

#
# Use NumPy arrays for creating a 3-dimensional array of variables,
# then use it to create a mode.
#

from __future__ import print_function

import numpy  as np
import xpress as xp

S1 = range (2)
S2 = range (3)
S3 = range (4)

m = xp.problem ()

h = np.array ([[[xp.var (vartype = xp.binary) for i in S1] for j in S2] for k in S3])

m.addVariable (h)
	
m.setObjective  (h[0][0][0] * h[0][0][0] + 
                 h[1][0][0] * h[0][0][0] +
                 h[1][0][0] * h[1][0][0] +
                 xp.Sum (h[i][j][k] for i in S3 for j in S2 for k in S1))

cons00 = - h[0][0][0] * h[0][0][0] + xp.Sum (i * j * k * h[i][j][k] for i in S3 for j in S2 for k in S1) >= 11

m.addConstraint (cons00)

m.solve ()

# Get the matrix representation of the quadratic part of the single
# constraint

mstart1=[]
mclind1=[]
dqe1=[]
m.getqrowqmatrix (cons00, mstart1, mclind1, dqe1, 29, h[0][0][0], h[3][2][1])
print ("row 0:", mstart1, mclind1, dqe1)
