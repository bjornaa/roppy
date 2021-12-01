# Example script for experimenting with different settings for vertical stretching
# Plots the depths of the mid-points of the vertical levels, aka rho-points.

import numpy as np
import matplotlib.pyplot as plt

import roppy

ξmax = 200  # Number of horizontal grid cells
Hmin = 50  # Minimum depth
Hmax = 1000  # Maximum depth
Hc = 50  # Critical depth

N = 60  # Number of s-levels
Θs = 5  # Surface stretching parameter
Θb = 3  # Bottom stretching parameter

Vtransform = 2
Vstretching = 5

ξ = np.arange(ξmax)  # Horizontal xi-coordinate

# Depth profile, half Gaussian seamount, standard deviation =  ξmax / sqrt(8)
H = Hmax - (Hmax - Hmin) * np.exp(-((2 * ξ / ξmax) ** 2))

# Compute the vertical stretching
C = roppy.s_stretch(N, Θs, Θb, Vstretching=Vstretching)

# Compute the depth of the sigma levels
S = roppy.sdepth(H, Hc, C, Vtransform=Vtransform)

# Plot the sigma levels
for k in range(N):
    plt.plot(ξ, S[k, :])

# Plot the seamount
plt.fill_between(ξ, -H, -Hmax, facecolor="lightgrey")
plt.plot(ξ, -H, color="black", lw=2)

# Axis text
plt.xlabel("ξ")
plt.ylabel("z")

# Limit the plot
plt.ylim(-Hmax, 0)
plt.xlim(0, ξmax)

plt.show()
