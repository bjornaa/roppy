# -*- coding: utf-8 -*-

"""
levels.py

A nice number is here defined as a number of form
a*10**e, where e is an integer and a is 1, 2, or 5

This module has utility functions for handling nice numbers,

  nice(x) 
      Returns closest nice number to x
  nice_levels(fmin, fmax, n=6)
      Returns a sequence of approximately n nice numbers
      between fmin and fmax

"""

# ----------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# 2010-12-20
# ----------------------------------


import math

 

def nice(v):
    """Returns nearest nice number

    Give highest value if equally close to two nice numbers

    """
    e = math.floor(math.log10(v))  # Exponent
    b = v / 10**e
    if b < 1.5:
        a = 1
    elif b < 3.5:
        a = 2
    elif b < 7.5:
        a = 5
    else:
        a = 10
    d = a*10**e
    return d

# Alternative (and slower) version
# Gives lover value if equally close to two nice numbers
def nice2(v):
    e = math.floor(math.log10(v))
    B = [10**e, 2*10**e, 5*10**e, 10*10**e, 20*10**e, 50*10**e, 100*10**e]
    d0, i0 = min([(abs(b-v),i) for i,b in enumerate(B)])
    return B[i0]
    

    

def nice_levels(fmin, fmax, n=6):
    """Returns approx. n nice equidistant levels between fmin and fmax

    arguments:
    fmin, fmax : floats, must have fmin < fmax
    n : hint for output sequence length, default n = 6

    returns:
    L : list of levels

    equidistance d is a nice number (a*10**e, a = 1, 2, or 5)
    returns a list L with L[i] = l0 + i*d, i = 1,...,m-1 with
    
    l0-d < fmin <= l0 = L[0] and L[-1] = l0+(m-1)*d <= fmax < l0 + m*d

    with m approximately equal to m

    """
    
    v = (fmax - fmin)/float(n)    # first guess at step length
    d = nice(v)                   # nice step length
    t0 = int(math.ceil(fmin/d))   # 
    t1 = int(math.floor(fmax/d))
    return [d*s for s in range(t0,t1+1)]

    
