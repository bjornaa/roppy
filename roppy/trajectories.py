# -*- coding: utf-8 -*-

# ------------------------------
# trajectories.py
#
# Particle tracking module for current field
# not recognising changes with time.
# Useful for streamlines and curly vectors
#
# Bjørn Ådlandsvik <bjorn@imr.no>
# 2010-03-03
# ------------------------------

# -------------
# Imports
# -------------

import numpy as np
import roppy

# -------------
# Functions
# -------------


def movepart(grid, U, V, X0, Y0, dt, nstep, order=4):
    """Move particles in a 2D current field

    grid   : roppy SGrid object
    U, V   : 2D Current field
    X0, Y0 : 1D arrays with grid coordinates of start positions
    dt     : timestep [seconds]
    nstep  : number of time steps

    Move particles in a time independent horizontal current field,
    grid must (presently) be a subgrid (i0, i1, j0, j1)
    U must be sliced to the grid
    Domain is limited by: i0 <= x < i1-1 and j0 <= y < j1-1
    Particles outside the domain are not moved

"""
    i0, i1, j0, j1 = grid.i0, grid.i1, grid.j0, grid.j1

    # Initalize
    X = np.zeros((nstep, len(X0)))
    Y = np.zeros((nstep, len(X0)))
    X[0, :] = X0
    Y[0, :] = Y0

    # Particle tracking loop

    # Euler-forward
    if order == 1:
        for t in range(nstep-1):
            pm = roppy.sample2D(grid.pm, X[t, :]-i0, Y[t, :]-j0,
                                outside_value=0)
            pn = roppy.sample2D(grid.pn, X[t, :]-i0, Y[t, :]-j0,
                                outside_value=0)
            Ut = roppy.sample2D(U, X[t, :]-i0+0.5, Y[t, :]-j0,
                                outside_value=0)
            Vt = roppy.sample2D(V, X[t, :]-i0, Y[t, :]-j0+0.5,
                                outside_value=0)
            X[t+1, :] = X[t, :] + Ut*dt*pm
            Y[t+1, :] = Y[t, :] + Vt*dt*pn

    # Runge-Kutta 4th order
    elif order == 4:
        for t in range(nstep-1):
            pm = roppy.sample2D(grid.pm, X[t, :]-i0, Y[t, :]-j0,
                                outside_value=0)
            pn = roppy.sample2D(grid.pn, X[t, :]-i0, Y[t, :]-j0,
                                outside_value=0)
            x0 = X[t, :] - i0
            y0 = Y[t, :] - j0
            dx1 = pm * dt * roppy.sample2D(U, x0+0.5, y0,
                                           outside_value=0)
            dy1 = pm * dt * roppy.sample2D(V, x0, y0+0.5,
                                           outside_value=0)
            dx2 = pm * dt * roppy.sample2D(U, x0+0.5*dx1+0.5, y0+0.5*dy1,
                                           outside_value=0)
            dy2 = pm * dt * roppy.sample2D(V, x0+0.5*dx1, y0+0.5*dy1+0.5,
                                           outside_value=0)
            dx3 = pm * dt * roppy.sample2D(U, x0+0.5*dx2+0.5, y0+0.5*dy2,
                                           outside_value=0)
            dy3 = pm * dt * roppy.sample2D(V, x0+0.5*dx2, y0+0.5*dy2+0.5,
                                           outside_value=0)
            dx4 = pm * dt * roppy.sample2D(U, x0+dx3+0.5, y0+dy3,
                                           outside_value=0)
            dy4 = pm * dt * roppy.sample2D(V, x0+dx3, y0+dy3+0.5,
                                           outside_value=0)
            dx = (dx1 + 2*dx2 + 2*dx3 + dx4)/6.0
            dy = (dy1 + 2*dy2 + 2*dy3 + dy4)/6.0
            X[t+1, :] = X[t, :] + dx
            Y[t+1, :] = Y[t, :] + dy

    return X, Y

# ------------------------------------------------


def curly_vectors(grid, U, V, stride=5,
                  dt=3600, nstep=72, order=4):
    """Compute data for curly vectors a.k.a. sperm plot

    grid : SGrid object
    U, V : 2d velocity components (C-grid)
    stride : distance between vectors
    dt : "time" step [s]
    nstep : Number of time steps
    order : 1 or 4 : Order of numerical method

    Returns X, Y
    Arrays of trajectories, shape = (nstep, N) where
    N = number of curly vectors
    """

    # Set up start points
    X, Y = np.meshgrid(grid.X, grid.Y)
    M = grid.mask_rho
    # Take the strides and flatten the arrays
    X = X[1:-1:stride, 1:-1:stride].ravel()
    Y = Y[1:-1:stride, 1:-1:stride].ravel()
    M = M[1:-1:stride, 1:-1:stride].ravel()
    # Use only sea points
    X = X[M > 0.5]
    Y = Y[M > 0.5]

    return movepart(grid, U, V, X, Y,
                    dt=dt, nstep=nstep, order=order)

# -----------


def update_trajectories(grid, X, Y, U, V, dt=3600, nstep=6):
    """Update trajectories

    X, Y : 2D arrays (time, id) of a partial trajectoris
    U, V : 2D horizontal current field
    """

    # Note thar movepart, returns initital as first.
    A, B = movepart(grid, U, V, X[-1], Y[-1], dt, nstep+1)
    X = np.concatenate((X[nstep:, :], A[1:, :]))
    Y = np.concatenate((Y[nstep:, :], B[1:, :]))
    return X, Y

# ---------------------------------------------


def arrow_head(X, Y, length):
    """Return grid coordinates of arrow heads

    X, Y : 2D arrays of curly paths
           axis=0, vertices (or time) along the tracks
           axis=1, track index

    Returns:
    xhead, yhead: 2D arrays of shape (3,X.shape[1])
           axis=0, first leg, tip and second leg of head
           axis=1, track index
    """

    # Useful values for length of arrow head depend on
    # resolution and zoom factor
    # May look strange for small vectors that curl a lot

    # Arrow head
    alpha = 20  # Angle [degrees]
    cosa = np.cos(alpha*np.pi/180.0)
    sina = np.sin(alpha*np.pi/180.0)

    # Arrow direction determined by
    # average of last 4 segments
    # number of segments should depend on resolution
    A = X[-1, :] - X[-5, :]
    B = Y[-1, :] - Y[-5, :]
    D = np.sqrt(A*A + B*B)   # Normalizing factor
    # Set up coordinates for
    xhead = np.outer(np.ones(3), X[-1, :])
    yhead = np.outer(np.ones(3), Y[-1, :])
    xhead[0, :] += -A*length*cosa/D - B*length*sina/D
    yhead[0, :] += -B*length*cosa/D + A*length*sina/D
    xhead[2, :] += -A*length*cosa/D + B*length*sina/D
    yhead[2, :] += -B*length*cosa/D - A*length*sina/D

    return xhead, yhead
