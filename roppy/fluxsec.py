# -*- coding: utf-8 -*-

# Section class for flux calculations

import math
import numpy as np



def staircase_from_line(i0, i1, j0, j1):

    swapXY = False
    if abs(i1-i0) < abs(j0-j1): # Mostly vertical
        i0, i1, j0, j1 = j0, j1, i0, i1
        swapXY = True    

    # Find integer points X0 and Y0 on line
    if i0 < i1:
        X0 = list(range(i0, i1+1))
    elif i0 > i1:
        X0 = list(range(i0, i1-1, -1))
    else:  # i0 = i1 and j0 = j1
        raise ValueError, "Section reduced to a point"
    slope = float(j1-j0) / (i1-i0)
    Y0 = [j0 + slope*(x-i0) for x in X0]

    # sign = -1 if Y0 is decreasing, otherwise sign = 1
    sign = 1
    if Y0[-1] < Y0[0]: sign = -1   # Decreasing Y

    # Make lists of positions along staircase
    X, Y = [i0], [j0]

    for i in range(len(X0)-1):
        x, y = X[-1], Y[-1]          # Last point on list
        x0, y0 = X0[i], Y0[i]        # Present point along line
        x1, y1 = X0[i+1], Y0[i+1]    # Next point along line
        if abs(y - y1) > 0.5:        # Append extra point
            if sign*(y - y0) < 0:        # Jump first
                X.append(x0)
                Y.append(y+sign)
                X.append(x1)
                Y.append(y+sign)
            else:                        # Jump last
                X.append(x1)
                Y.append(y)
                X.append(x1)
                Y.append(y+sign)
        else:                        # Ordinary append
            X.append(x1)
            Y.append(y)

    if swapXY:
        X, Y = Y, X

    return np.array(X), np.array(Y)


    ## if abs(i1-i0) >= abs(j0-j1): # Work horizontally
        

    ##     # Find integer points X0 and Y0 on line
    ##     if i0 < i1:
    ##         X0 = list(range(i0, i1+1))
    ##     elif i0 > i1:
    ##         X0 = list(range(i0, i1-1, -1))
    ##     else:  # i0 = i1 and j0 = j1
    ##         raise ValueError, "Section reduced to a point"
    ##     slope = float(j1-j0) / (i1-i0)
    ##     Y0 = [j0 + slope*(x-i0) for x in X0]

    ##     # sign = -1 if Y0 is decreasing, otherwise sign = 1
    ##     sign = 1
    ##     if Y0[-1] < Y0[0]: sign = -1   # Decreasing Y

    ##     # Make lists of positions along staircase
    ##     X = [i0]
    ##     Y = [j0]  # Start

    ##     for i in range(len(X0)-1):
    ##         x, y = X[-1], Y[-1]          # Last point on list
    ##         x0, y0 = X0[i], Y0[i]        # Present point along line
    ##         x1, y1 = X0[i+1], Y0[i+1]    # Next point along line
    ##         if abs(y - y1) > 0.5:        # Append extra point
    ##             if sign*(y - y0) < 0:        # Jump first
    ##                 X.append(x0)
    ##                 Y.append(y+sign)
    ##                 X.append(x1)
    ##                 Y.append(y+sign)
    ##             else:                        # Jump last
    ##                 X.append(x1)
    ##                 Y.append(y)
    ##                 X.append(x1)
    ##                 Y.append(y+sign)
    ##         else:                      # Ordinary append
    ##             X.append(x1)
    ##             Y.append(y)

        
    ## else:  # Work in Y-direction
    ##     if j0 < j1:
    ##         Y0 = list(range(j0, j1+1))
    ##     elif j0 > j1:
    ##         Y0 = list(range(j0, j1-1, -1))
    ##     slope = float(i1-i0) / (j1-j0)
    ##     X0 = [i0 + slope*(y-j0) for y in Y0]

    ##     sign = 1
    ##     if X0[-1] < X0[0]: sign = -1   # Decreasing X
    ##     X = [X0[0]]
    ##     Y = [Y0[0]]  # Start

    ##     for i in range(len(X0)-1):
    ##         x, y = X[-1], Y[-1]
    ##         x0, y0 = X0[i], Y0[i]
    ##         x1, y1 = X0[i+1], Y0[i+1]
    ##         if abs(x - x1) > 0.5:      #  Append extra point
    ##             if sign*(x - x0) < 0:       # Jump first
    ##                 X.append(x+sign)
    ##                 Y.append(y0)
    ##                 X.append(x+sign)
    ##                 Y.append(y1)
    ##             else:                 # Jump last
    ##                 X.append(x)
    ##                 Y.append(y1)
    ##                 X.append(x+sign)
    ##                 Y.append(y1)
    ##         else:                 # Ordinary append
    ##             X.append(x)
    ##             Y.append(y1)

    return np.array(X), np.array(Y)


  
