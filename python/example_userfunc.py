#
# Python equivalent of the ComplexUserFunction.c example in examples/nonlinear/c directory
#

# Define objective and constraint as user functions that return
# derivatives
#
# Minimize myobj (y,z,v,w)
#   s.t.
#   ball (x,t) <= 730
#   1 <= x <= 2
#   2 <= y <= 3
#   3 <= z <= 4
#   4 <= v <= 5
#   5 <= w <= 6
#
# where
#
# myobj (y,z,v,w) = y**2 + z - v + w**2
# ball  (x,t) = x**2 + t**2

import xpress

def myobj (y,z,v,w):
    return (y**2 + z - v + w**2, # value of the function
            2*y,                 # derivatives w.r.t. y
            1,                   #                    z
            -1,                  #                    v
            2*w)                 #                    w

def ball (x,t):
    return (x**2 + t**2,
            2*x,
            2*t)

x = xpress.var (lb = 1, ub = 2)
y = xpress.var (lb = 2, ub = 3)
z = xpress.var (lb = 3, ub = 4)
v = xpress.var (lb = 4, ub = 5)
w = xpress.var (lb = 5, ub = 6)

t = xpress.var (lb = -xpress.infinity) # free variable

p = xpress.problem ()

p.addVariable (x,y,z,v,w,t)

p.setObjective (t)
p.addConstraint (t == xpress.user (myobj, y,z,v,w))
p.addConstraint (xpress.user (ball, x, t) <= 730)

p.solve ()

print ('objective:', p.getObjVal(), '; solution:', p.getSolution ())
