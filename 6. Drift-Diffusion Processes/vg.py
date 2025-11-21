import numpy as np
import matplotlib.pyplot as plt

# %%
# Simulate a time-changed arithmetic Brownian motion: the Variance Gamma process
# dX(t) = mu * dG(t) + sigma * dW(G(t))

# Parameters
npaths = 20000
T = 1.0
nsteps = 200
dt = T / nsteps
t = np.linspace(0, T, nsteps + 1)

mu = 0.2
sigma = 0.3
kappa = 0.05  # scale parameter of the Gamma process

# %%
# Monte Carlo simulation

# Gamma increments: MATLAB gamrnd(shape, scale)
# shape = dt/kappa, scale = kappa
dG = np.random.gamma(shape=dt/kappa, scale=kappa, size=(nsteps, npaths))

# ABM increments evaluated on randomised clock
dX = mu * dG + sigma * np.random.randn(nsteps, npaths) * np.sqrt(dG)

# Accumulate
X = np.vstack([np.zeros((1, npaths)), np.cumsum(dX, axis=0)])

# %%
# Expected, mean and sample path

plt.figure(figsize=(10, 5))

EX = mu * t
plt.plot(t, EX, "k", label="Expected path")
plt.plot(t, X.mean(axis=1), ":k", label="Mean path")
plt.plot(t, X[:, ::1000])  # sample paths, every 1000th

plt.xlabel("t")
plt.ylabel("X")
plt.ylim([-0.8, 1.2])
plt.title("Paths of a Variance Gamma process  dX(t)=μ dG(t)+σ dW(G(t))")
plt.legend()
plt.show()

# %%
# Probability densities at different times

plt.figure(figsize=(8, 10))
bins = 100

# t = 0.2  (index 40)
h, x = np.histogram(X[40, :], bins=bins, density=True)
xc = 0.5 * (x[:-1] + x[1:])
plt.subplot(3, 1, 1)
plt.bar(xc, h, width=(x[1]-x[0]))
plt.ylabel("$f_X(x, 0.2)$")
plt.xlim([-0.8, 1.2])
plt.ylim([0, 6])
plt.title("Probability density function of a Variance Gamma process")

# t = 0.5  (index 100)
h, x = np.histogram(X[100, :], bins=bins, density=True)
xc = 0.5 * (x[:-1] + x[1:])
plt.subplot(3, 1, 2)
plt.bar(xc, h, width=(x[1]-x[0]))
plt.ylabel("$f_X(x, 0.5)$")
plt.xlim([-0.8, 1.2])
plt.ylim([0, 3])

# t = 1.0  (final)
h, x = np.histogram(X[-1, :], bins=bins, density=True)
xc = 0.5 * (x[:-1] + x[1:])
plt.subplot(3, 1, 3)
plt.bar(xc, h, width=(x[1]-x[0]))
plt.xlabel("x")
plt.ylabel("$f_X(x, 1)$")
plt.xlim([-0.8, 1.2])
plt.ylim([0, 3])

plt.show()