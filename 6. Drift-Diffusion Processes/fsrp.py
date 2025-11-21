# Monte Carlo simulation of the Feller square-root process (CIR)
# dX = alpha*(mu - X) dt + sigma*sqrt(X) dW
# Used in the Cox–Ingersoll–Ross model and Heston stochastic volatility model

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.stats import ncx2

# ==== Reusable CIR utilities ====
def cir_mean(t, X0, mu, alpha):
    """E[X_t] for CIR."""
    t = np.asarray(t, dtype=float)
    return mu + (X0 - mu) * np.exp(-alpha * t)

def cir_long_run_std(sigma, mu, alpha):
    """Long-run standard deviation of CIR."""
    return sigma * np.sqrt(mu / (2.0 * alpha))

def cir_feller_ratio(alpha, mu, sigma):
    """Feller ratio 2*alpha*mu/sigma^2."""
    return 2.0 * alpha * mu / (sigma ** 2)

def cir_em_moment_coeffs(alpha, mu, sigma, dt):
    """
    Coefficients for analytic-moments Euler step:
    X_{t+dt} ≈ mu + (X_t - mu) e^{-alpha dt} + sqrt(a X_t + b) * N
    """
    a = (sigma ** 2 / alpha) * (np.exp(-alpha * dt) - np.exp(-2.0 * alpha * dt))
    b = (mu * sigma ** 2 / (2.0 * alpha)) * (1.0 - np.exp(-alpha * dt)) ** 2
    return a, b

def cir_em_step(x, n, alpha, mu, sigma, dt, a=None, b=None):
    """
    One-step update using analytic moments with full truncation.
    x: current state array, n: normal draws matching x.shape
    a, b: optional precomputed coefficients from cir_em_moment_coeffs
    """
    if a is None or b is None:
        a, b = cir_em_moment_coeffs(alpha, mu, sigma, dt)
    drift = mu + (x - mu) * np.exp(-alpha * dt)
    vol_term = np.sqrt(np.maximum(a * x + b, 0.0)) * n
    x_next = drift + vol_term
    return np.maximum(x_next, 0.0)

def cir_ncx2_params(t, X0, alpha, mu, sigma):
    """
    Parameters (k, d, lambda) for the exact ncx2 transition of CIR at time t.
    t can be a scalar or array.
    """
    t = np.asarray(t, dtype=float)
    k = (sigma ** 2) * (1.0 - np.exp(-alpha * t)) / (4.0 * alpha)
    d = 4.0 * alpha * mu / (sigma ** 2)
    lam = 4.0 * alpha * X0 / (sigma ** 2 * (np.exp(alpha * t) - 1.0))
    return k, d, lam
# ==== End utilities ====

# %% ----- Parameters and time grid -----
npaths = 2000          # number of paths
T = 1.0                # time horizon
nsteps = 200           # number of time steps
dt = T / nsteps        # time step
t = np.linspace(0.0, T, nsteps + 1)  # observation times

alpha = 5.0
mu = 0.07
sigma = 0.265
# alpha, mu, sigma = 5.0, 0.03, 0.8

X0 = 0.03
Feller_ratio = cir_feller_ratio(alpha, mu, sigma)
print("Feller ratio:", Feller_ratio)

# Optional reproducibility
# rng = np.random.default_rng(42)
# N = rng.standard_normal(size=(nsteps, npaths))

# %% ----- Monte Carlo: allocate and initialise all paths -----
X = np.empty((nsteps + 1, npaths), dtype=float)
X[0, :] = X0

# Euler–Maruyama with analytic moments and full truncation
# Precompute normal draws and coefficients
N = np.random.standard_normal(size=(nsteps, npaths))
a, b = cir_em_moment_coeffs(alpha, mu, sigma, dt)

start = perf_counter()
for i in range(nsteps):
    X[i + 1, :] = cir_em_step(X[i, :], N[i, :], alpha, mu, sigma, dt, a=a, b=b)
elapsed = perf_counter() - start
print(f"Simulation time: {elapsed:.4f} s")

# %% ----- Expected, mean and sample paths -----
EX = cir_mean(t, X0, mu, alpha)
sdev_infty = cir_long_run_std(sigma, mu, alpha)

plt.figure(1)
# plot a subset of paths to avoid clutter
step = max(npaths // 25, 10)
plt.plot(t, X[:, ::step], linewidth=0.5, alpha=0.6)
# Expected path, mean path, long-run mean, and a subset of sample paths
plt.plot(t, EX, 'rx', label='Expected path')
plt.plot(t, X.mean(axis=1), ':k', label='Mean path')
plt.plot(t, mu * np.ones_like(t), 'k--', label=r'$\mu$')

plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('X')
plt.ylim([-0.02, mu + 4 * sdev_infty])
plt.title(r'Paths of a Feller square-root process $dX=\alpha(\mu-X)\,dt+\sigma X^{1/2} dW$')
plt.tight_layout()
# plt.savefig('fsrppaths.png', dpi=300)
plt.show()

# %% ----- Probability density function at different times -----
t2 = np.array([0.05, 0.10, 0.20, 0.40, 1.00])
x = np.linspace(-0.02, mu + 4 * sdev_infty, 200)

# Analytic ncx2 parameters
k, d, lam = cir_ncx2_params(t2, X0, alpha, mu, sigma)

fa = np.zeros((x.size, t2.size))  # analytical
fs = np.zeros((x.size, t2.size))  # sampled histogram-based density

# Precompute histogram bin edges to mimic MATLAB hist(x with centres)
dx = x[1] - x[0]
edges = np.concatenate(([x[0] - dx / 2], x[:-1] + dx / 2, [x[-1] + dx / 2]))

for j, tj in enumerate(t2):
    # Analytic density: f_X(x,t) = (1/k) * f_ncx2(x/k; d, lambda)
    # SciPy's ncx2.pdf uses df = d, nc = lam[j]
    # Guard: for x < 0, density is zero since CIR state is non-negative
    z = np.clip(x / k[j], a_min=0.0, a_max=None)
    fa[:, j] = ncx2.pdf(z, d, lam[j]) / k[j]

    # Sampled density at time index closest to tj
    idx = int(round(tj / dt))
    idx = min(max(idx, 0), nsteps)

    counts, _ = np.histogram(X[idx, :], bins=edges)
    fs[:, j] = counts / (npaths * dx)  # convert counts to density at centres

plt.figure(2)
for j, tj in enumerate(t2):
    plt.plot(x, fa[:, j], label=f't = {tj:0.2f} (analytic)')
    plt.plot(x, fs[:, j], linestyle='--', label=f't = {tj:0.2f} (sampled)')

plt.axvline(x=X0, color='b', linestyle=':', alpha=0.3, label=r'$X_0$')
plt.axvline(x=mu, color='k', linestyle=':', alpha=0.3, label=r'$\mu$')
plt.xlabel('x')
plt.ylabel(r'$f_X(x,t)$')
plt.title('Probability density function of a Feller square-root process at different times')
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
# plt.savefig('fsrpdensities.png', dpi=300)
plt.show()