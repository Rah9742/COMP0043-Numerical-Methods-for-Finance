import numpy as np
import matplotlib.pyplot as plt
import time

# %% Monte Carlo simulation of the Ornstein Uhlenbeck process
# dX = alpha*(mu - X)*dt + sigma*dW

# Parameters and time grid
npaths = 20000
T = 1.0
nsteps = 200
dt = T / nsteps
t = np.linspace(0.0, T, nsteps + 1)  # shape (nsteps+1,)
alpha = 5.0
mu = 0.07
sigma = 0.07
X0 = 0.03

# %% Monte Carlo

# Allocate and initialise all paths
# MATLAB did X = [X0*ones(1,npaths); zeros(nsteps,npaths)]
# which is (nsteps+1, npaths)
X = np.zeros((nsteps + 1, npaths))
X[0, :] = X0

# Sample standard Gaussian random numbers
N = np.random.randn(nsteps, npaths)

start = time.perf_counter()

# Standard deviation for a time step
# plain Euler Maruyama would be: sdev = sigma * np.sqrt(dt)
# here we use Euler Maruyama with analytic moments
sdev = sigma * np.sqrt((1 - np.exp(-2 * alpha * dt)) / (2 * alpha))

# Compute and accumulate the increments
exp_term = np.exp(-alpha * dt)
for i in range(nsteps):
    # Plain EM would be:
    # X[i+1, :] = X[i, :] + alpha * (mu - X[i, :]) * dt + sdev * N[i, :]
    # With analytic moments:
    X[i + 1, :] = mu + (X[i, :] - mu) * exp_term + sdev * N[i, :]

end = time.perf_counter()
print(f"Simulation time: {end - start:.4f} seconds")

# %% Expected, mean and sample paths, long term average
plt.figure(1)
EX = mu + (X0 - mu) * np.exp(-alpha * t)
plt.plot(t, EX, 'k')
plt.plot(t, X.mean(axis=1), 'rx', label='Mean path')
plt.plot(t, mu * np.ones_like(t), 'k--', label='Long term average')
# sample some paths
plt.plot(t, X[:, ::1000], linewidth=0.7)
plt.plot(t, EX, 'k', label='Expected path')
plt.plot(t, X.mean(axis=1), 'k:')
plt.plot(t, mu * np.ones_like(t), 'k--')

sdev_infty = sigma / np.sqrt(2 * alpha)
plt.ylim([mu - 4 * sdev_infty, mu + 4 * sdev_infty])
plt.xlabel('t')
plt.ylabel('X')
plt.title(r'Ornstein Uhlenbeck process $dX = \alpha(\mu - X)\,dt + \sigma\,dW$')
plt.legend()
plt.tight_layout()
# plt.savefig('oupaths.png', dpi=300)
plt.show()

# %% Variance = mean square deviation
plt.figure(2)
theory_var = sigma**2 / (2 * alpha) * (1 - np.exp(-2 * alpha * t))
plt.plot(t, theory_var, 'r', label='Theory')
plt.plot(t, sigma**2 * t, 'g', label=r'$\sigma^2 t$')
plt.plot(t, sigma**2 / (2 * alpha) * np.ones_like(t), 'b', label=r'$\sigma^2/(2\alpha)$')
plt.plot(t, np.var(X, axis=1), 'm', label='Sampled 1')
plt.plot(t, np.mean((X - EX[:, None])**2, axis=1), 'c--', label='Sampled 2')
plt.xlabel('t')
plt.ylabel(r'Var(X) = E((X - E(X))^2)')
plt.ylim([0, 0.0006])
plt.legend(loc='lower right')
plt.title('Ornstein Uhlenbeck process: variance')
plt.tight_layout()
# plt.savefig('ouvariance.png', dpi=300)
plt.show()

# %% Mean absolute deviation
plt.figure(3)
theory_mad = sigma * np.sqrt((1 - np.exp(-2 * alpha * t)) / (np.pi * alpha))
plt.plot(t, theory_mad, 'r', label='Theory')
plt.plot(t, sigma * np.sqrt(2 * t / np.pi), 'g', label=r'$\sigma (2t/\pi)^{1/2}$')
plt.plot(t, sigma / np.sqrt(np.pi * alpha) * np.ones_like(t), 'b', label='Long term average')
plt.plot(t, np.mean(np.abs(X - EX[:, None]), axis=1), 'm', label='Sampled')
plt.xlabel('t')
plt.ylabel(r'$E(|X - E(X)|) = (2\,Var(X)/\pi)^{1/2}$')
plt.ylim([0, 0.02])
plt.legend(loc='lower right')
plt.title('Ornstein Uhlenbeck process: mean absolute deviation')
plt.tight_layout()
# plt.savefig('mad.png', dpi=300)
plt.show()

# %% Probability density function at different times
x = np.linspace(-0.02, mu + 4 * sdev_infty, 200)   # 200 edges
dx = x[1] - x[0]

t2 = np.array([0.05, 0.1, 0.2, 0.4, 1.0])
EX2 = mu + (X0 - mu) * np.exp(-alpha * t2)
sdev2 = sigma * np.sqrt((1 - np.exp(-2 * alpha * t2)) / (2 * alpha))

# fa and fs must both have length 199 now
fa = np.zeros((len(x) - 1, len(t2)))
fs = np.zeros((len(x) - 1, len(t2)))

for i in range(len(t2)):
    # analytical pdf evaluated at bin centres
    x_centres = 0.5 * (x[:-1] + x[1:])
    fa[:, i] = 1.0 / (np.sqrt(2 * np.pi) * sdev2[i]) * np.exp(-(x_centres - EX2[i])**2 / (2 * sdev2[i]**2))

    idx = int(t2[i] * nsteps)
    hist_counts, _ = np.histogram(X[idx, :], bins=x)
    fs[:, i] = hist_counts / (npaths * dx)

plt.figure(4)
for i in range(len(t2)):
    plt.plot(x_centres, fa[:, i])
    plt.plot(x_centres, fs[:, i])
plt.legend(['t = 0.05', 't = 0.05 sampled',
            't = 0.10', 't = 0.10 sampled',
            't = 0.20', 't = 0.20 sampled',
            't = 0.40', 't = 0.40 sampled',
            't = 1.00', 't = 1.00 sampled'],
           ncol=2)
plt.xlabel('x')
plt.ylabel(r'$f_X(x,t)$')
plt.title('Ornstein Uhlenbeck process: PDF at different times')
plt.tight_layout()
# plt.savefig('oudensities.png', dpi=300)
plt.show()

# %% Autocovariance
# We follow your MATLAB logic: xcorr per path, unbiased, then average
n = nsteps + 1  # length of each path
C = np.zeros((2 * n - 1, npaths))

denom = np.concatenate((np.arange(1, n), np.arange(n, 0, -1)))  # for unbiased

for j in range(npaths):
    xj = X[:, j] - EX
    cj = np.correlate(xj, xj, mode='full') / denom
    C[:, j] = cj

C_mean = C.mean(axis=1)

plt.figure(5)
tau = t  # lags for non negative part
theory_cov = sigma**2 / (2 * alpha) * np.exp(-alpha * tau)
plt.plot(tau, theory_cov, 'r', label='Theory for infinite t')
plt.plot(tau, C_mean[n - 1:], 'g', label='Sampled')
plt.plot(0, sigma**2 / (2 * alpha), 'go', label='Var for infinite t')
plt.plot(0, np.mean(np.var(X, axis=1)), 'bo', label='Sampled Var')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$C(\tau)$')
plt.title('Ornstein Uhlenbeck process: autocovariance')
plt.legend()
plt.tight_layout()
# plt.savefig('ouautocov.png', dpi=300)
plt.show()

# %% Autocorrelation
plt.figure(6)
plt.plot(tau, np.exp(-alpha * tau), 'r', label='Theory for infinite t')
plt.plot(tau, C_mean[n - 1:] / C_mean[n - 1], 'g', label='Sampled')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$c(\tau)$')
plt.title('Ornstein Uhlenbeck process: autocorrelation')
plt.legend()
plt.tight_layout()
# plt.savefig('ouautocorr.png', dpi=300)
plt.show()
