# %% Acceptance-rejection method for Gaussian from Laplace random variables

import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 10**6        # number of samples
xmax = 4.0       # grid bound
deltax = 0.2     # grid step
x = np.arange(-xmax, xmax + 1e-12, deltax)  # grid

# Densities
f = lambda x: (1 / np.sqrt(2 * np.pi)) * np.exp(-(x**2) / 2)      # standard normal PDF
g = lambda x: 0.5 * np.exp(-np.abs(x))                             # standard Laplace PDF

# Optimal c: f = c g at x = ±1, so c = max_x f/g = sqrt(2e/pi)
c = np.sqrt(2 * np.e / np.pi)

# --- Sample the standard Laplace (double-sided exponential) via inverse CDF ---
U1 = np.random.rand(n)

L = np.where(U1 < 0.5, np.log(2 * U1), -np.log(2 * (1 - U1)))  # location 0, scale 1
print(U1)

# --- Acceptance-Rejection to sample standard normal using Laplace proposal ---
gL = g(L)
fL = f(L)
U2 = np.random.rand(n)
mask = U2 * c * gL <= fL
N = L[mask]  # accepted samples

# --- Console outputs ---
acceptance_ratio = N.size / n
print("Acceptance ratio:", acceptance_ratio)
print("c (analytical):", c)
print("max(f(L)/g(L)) on the proposal samples:", np.max(fL / gL))

# # --- Figure 1: Histogram of accepted samples vs theoretical curves ---
# plt.figure(figsize=(8, 5))
#
# # Build bin edges like MATLAB's left-edge shift: edges from -xmax-Δ/2 to xmax+Δ/2
# edges = np.arange(-xmax - deltax/2, xmax + deltax/2 + 1e-12, deltax)
# plt.hist(N, bins=edges, density=True, alpha=0.6, label='Sampled f(x)')
#
# fx = f(x)
# gx = g(x)
# plt.plot(x, fx, label='Theoretical f(x)')
# plt.plot(x, c * gx, label='Majorant function c g(x)')
#
# plt.xlabel('x')
# plt.legend()
# plt.title('Standard normal distribution using the acceptance-rejection algorithm')
# plt.show()
#
# # --- Figure 2: Show where f/g attains its maximum c at x = 1 ---
# plt.figure(figsize=(8, 5))
# plt.plot(x, x**2 - 2*x + 1, label='x^2 - 2x + 1')
# plt.plot(x, fx / gx, label='f/g')
# plt.plot(x, c * np.ones_like(x), '--', label=r'c = $(2e/\pi)^{1/2}$')
# plt.xlim(0, 3)
# plt.xlabel('x')
# plt.legend(loc='upper left')
# plt.title('x^2 - 2x + 1 = 0 where f/g = max = c')
# plt.show()