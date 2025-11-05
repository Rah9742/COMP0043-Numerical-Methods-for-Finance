import numpy as np
import matplotlib.pyplot as plt

# %% Monte Carlo simulation of geometric Brownian motion
# dS = mu*S*dt + sigma*S*dW

# Define parameters and time grid
np.random.seed(0)
npaths = 20000   # number of paths
T = 1.0          # time horizon
nsteps = 200     # number of time steps
dt = T / nsteps  # time step
t = np.linspace(0, T, nsteps + 1)  # observation times
mu = 0.2
sigma = 0.4
S0 = 1.0

# %% Monte Carlo
# Compute increments of arithmetic Brownian motion X = log(S/S0)
dX = (mu - 0.5 * sigma**2) * dt + sigma * np.random.randn(npaths, nsteps) * np.sqrt(dt)

# Accumulate increments
X = np.concatenate([np.zeros((npaths, 1)), np.cumsum(dX, axis=1)], axis=1)

# Transform to geometric Brownian motion
S = S0 * np.exp(X)

# %% Expected, mean, and sample paths
plt.figure(figsize=(8, 5))
EX = np.exp(mu * t)  # expected path
plt.plot(t, EX, 'k', label='Expected path')
plt.plot(t, np.mean(S, axis=0), ':k', label='Mean path')
plt.plot(t, S[::1000, :].T, linewidth=0.7, alpha=0.6)
plt.ylim([0, 2.5])
plt.xlabel('t')
plt.ylabel('X')
plt.title(r'Geometric Brownian motion $dS = \mu S dt + \sigma S dW$')
plt.legend()
plt.tight_layout()
plt.savefig('gbppaths.png', dpi=300)
plt.show()

# %% Probability density function at different times
plt.figure(figsize=(6, 8))

plt.subplot(3, 1, 1)
plt.hist(S[:, 20], bins=np.arange(0, 3.5, 0.035), density=True)
plt.xlim([0, 3.5])
plt.ylim([0, 3.5])
plt.ylabel(r'$f_X(x, 0.15)$')
plt.title('Geometric Brownian motion: PDF at different times')

plt.subplot(3, 1, 2)
plt.hist(S[:, 80], bins=np.arange(0, 3.5, 0.035), density=True)
plt.xlim([0, 3.5])
plt.ylim([0, 3.5])
plt.ylabel(r'$f_X(x, 0.4)$')

plt.subplot(3, 1, 3)
plt.hist(S[:, -1], bins=np.arange(0, 3.5, 0.035), density=True)
plt.xlim([0, 3.5])
plt.ylim([0, 3.5])
plt.xlabel('x')
plt.ylabel(r'$f_X(x, 1)$')

plt.tight_layout()
plt.savefig('gbpdensities.png', dpi=300)
plt.show()