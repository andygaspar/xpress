#!/bin/env python

#
# Define the well-known Rosenbrock function and minimize it
#

from __future__ import print_function

import xpress as xp

x = xp.var (lb = -xp.infinity)
y = xp.var (lb = -xp.infinity)

p = xp.problem ()

p.addVariable (x, y)

# parameters of the Rosenbrock function
a = 1
b = 100

p.setObjective ((a - x)**2 + b * (y - x**2)**2)

p.solve ()

print ('solution: ', p.getSolution (), '; value: ', p.getObjVal ())
