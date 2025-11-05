import numpy as np
import matplotlib.pyplot as plt

# %% Define parameters and time grid
npaths = 20000  # number of paths
T = 1.0         # time horizon
nsteps = 200    # number of time steps
dt = T / nsteps
t = np.linspace(0, T, nsteps + 1)

mu = -0.05
sigma = 0.4

# %% Monte Carlo simulation of arithmetic Brownian motion
# dX = mu*dt + sigma*dW

# Euler–Maruyama increments

dX = mu * dt + sigma * np.sqrt(dt) * np.random.randn(npaths, nsteps)

# Accumulate increments
X = np.concatenate([np.zeros((npaths, 1)), np.cumsum(dX, axis=1)], axis=1)

# %% Expected, mean, and sample path
EX = mu * t
plt.figure(figsize=(8, 5))
plt.plot(t, EX, 'k', label='Expected path')
plt.plot(t, X[::1000].T, color='gray', alpha=0.3)
plt.plot(t, X.mean(axis=0), ':k', label='Mean path')
plt.xlabel('t')
plt.ylabel('X')
plt.ylim([-1, 1])
plt.legend()
plt.title(r'Arithmetic Brownian motion $dX_t = \mu dt + \sigma dW_t$')
plt.tight_layout()
# plt.savefig('abmpaths.png', dpi=300)
# plt.close()
plt.show()

# %% Variance (Mean Square Deviation)
plt.figure(figsize=(8, 5))
plt.plot(t, sigma**2 * t, 'k', label=r'Theory: $\sigma^2 t = 2Dt$')
plt.plot(t, X.var(axis=0), ':r', label='Sampled')
plt.xlabel('t')
plt.ylabel(r'Var(X) = E[(X - E[X])^2]')
plt.legend(loc='upper left')
plt.title('Arithmetic Brownian motion: MSD')
plt.tight_layout()
# plt.savefig('abmmsd.png', dpi=300)
# plt.close()
plt.show()

# %% Mean Absolute Deviation
plt.figure(figsize=(8, 5))
plt.plot(t, sigma * np.sqrt(2 * t / np.pi), 'k', label=r'Theory: $\sigma(2t/\pi)^{1/2}$')
plt.plot(t, np.mean(np.abs(X - EX), axis=0), ':r', label='Sampled')
plt.xlabel('t')
plt.ylabel(r'$E(|X - E[X]|) = (2Var(X)/\pi)^{1/2}$')
plt.ylim([0, 1.0])
plt.ylim([0, 0.35])
plt.legend(loc='upper left')
plt.title('Arithmetic Brownian motion: mean absolute deviation')
plt.tight_layout()
# plt.savefig('mad.png', dpi=300)
# plt.close()
plt.show()

# %% Probability density function at different times
plt.figure(figsize=(8, 8))

plt.subplot(3, 1, 1)
plt.hist(X[:, 20], bins=np.arange(-1, 1.02, 0.02), density=True)
plt.ylabel(r'$f_X(x, 0.1)$')
plt.xlim([-1, 1])
plt.ylim([0, 3.5])
plt.title('Arithmetic Brownian motion: PDF at different times')

plt.subplot(3, 1, 2)
plt.hist(X[:, 80], bins=np.arange(-1, 1.02, 0.02), density=True)
plt.ylabel(r'$f_X(x, 0.4)$')
plt.xlim([-1, 1])
plt.ylim([0, 3.5])

plt.subplot(3, 1, 3)
plt.hist(X[:, -1], bins=np.arange(-1, 1.02, 0.02), density=True)
plt.xlabel('x')
plt.ylabel(r'$f_X(x, 1)$')
plt.xlim([-1, 1])
plt.ylim([0, 3.5])

plt.tight_layout()
# plt.savefig('abmhist.png', dpi=300)
# plt.close()
plt.show()

# %% Solution of the Fokker–Planck equation
D = sigma**2 / 2
x_vals = np.arange(-1, 1.02, 0.02)
t_vals = np.arange(0.1, 1.025, 0.025)
x, tt = np.meshgrid(x_vals, t_vals)

f = (1 / (2 * np.sqrt(np.pi * D * tt))) * np.exp(-(x - mu * tt)**2 / (4 * D * tt))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, tt, f, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel(r'$f_X$')
ax.set_title(r'Arithmetic Brownian motion: Fokker–Planck solution ($\mu = -0.05$, $\sigma = 0.4$)')
ax.view_init(24, 30)
plt.tight_layout()
# plt.savefig('abmfpe.png', dpi=300)
# plt.close()
plt.show()
