# Example script for experimenting with different settings for vertical stretching

import numpy as np
import matplotlib.pyplot as plt

import roppy

ξmax = 200  # Number of horizontal grid cells
Hmin = 50  # Minimum depth
Hmax = 1000  # Maximum depth
Hc = 50  # Critical depth

N = 20  # Number of s-levels
Θs = 5  # Surface stretching parameter
Θb = 3  # Bottom stretching parameter

Vtransform = 2
Vstretching = 5

ξ = np.arange(ξmax)  # Horizontal xi-coordinate

# Depth profile, half Gaussian seamount
H = Hmax - (Hmax - Hmin) * np.exp(-((2 * ξ / ξmax) ** 2))

# Compute the vertical stretching
C = roppy.s_stretch(N, Θs, Θb, Vstretching=Vstretching)


# Compute the sigma coordinates
S = roppy.sdepth(H, Hc, C, Vtransform=Vtransform)

for k in range(N):
    plt.plot(ξ, S[k])

plt.plot(ξ, -H, color="black", lw=2)

plt.xlabel("ξ")
plt.ylabel("z")


plt.show()
