import numpy as np
import matplotlib.pyplot as plt

# %% ------------------------------------------------------------
# Monte Carlo simulation of the Heston model
# ------------------------------------------------------------

# Time grid and MC settings
T = 1.0
nsteps = 200
dt = T / nsteps
t = np.linspace(0, T, nsteps + 1)
npaths = 20000

# Market params
S0 = 1.0
r = 0.02

# Heston params
kappa = 3.0
Vbar = 0.1
sigmaV = 0.25        # vol of vol
V0 = 0.08
rho = -0.8

# Risk-neutral drift
mu = r + Vbar / 2.0

# Feller ratio
beta = kappa * Vbar / sigmaV**2
print("Feller ratio =", beta)

# ------------------------------------------------------------
# Monte Carlo Simulation
# ------------------------------------------------------------

# Generate correlated normal variates
N1 = np.random.randn(nsteps, npaths)
N2 = rho * N1 + np.sqrt(1 - rho**2) * np.random.randn(nsteps, npaths)

# Allocate variance and log-price arrays
V = np.zeros((nsteps + 1, npaths))
X = np.zeros((nsteps + 1, npaths))
S = np.zeros((nsteps + 1, npaths))

V[0, :] = V0
X[0, :] = 0.0      # since S0 = 1 → X = log(S/S0) = 0
S[0, :] = S0

# Analytic-moment coefficients for V update
a = sigmaV**2 / kappa * (np.exp(-kappa * dt) - np.exp(-2 * kappa * dt))
b = (
    Vbar * sigmaV**2 / (2 * kappa) * (1 - np.exp(-kappa * dt)) ** 2
)

# Euler–Maruyama for V with analytic-moment variant
for i in range(nsteps):

    # Plain Euler (in slides)
    # V[i+1, :] = V[i, :] + kappa*(Vbar - V[i, :])*dt + sigmaV*np.sqrt(V[i,:])*np.sqrt(dt)*N2[i,:]

    # === Euler–Maruyama variance update ===
    V[i+1, :] = V[i, :] + kappa * (Vbar - V[i, :]) * dt + sigmaV * np.sqrt(V[i, :]) * np.sqrt(dt) * N2[i, :]

    # === Analytic-moment corrected scheme (reference, commented) ===
    # V[i+1, :] = Vbar + (V[i, :] - Vbar) * np.exp(-kappa * dt) + \
    #             np.sqrt(a * V[i, :] + b) * N2[i, :]

    V[i+1, :] = np.maximum(V[i+1, :], 0.0)

    # Update X and S (Euler discretisation)
    X[i+1, :] = X[i, :] + (mu - 0.5 * V[i, :]) * dt + np.sqrt(V[i, :]) * np.sqrt(dt) * N1[i, :]
    S[i+1, :] = np.exp(X[i+1, :])

# %%
# ------------------------------------------------------------
# Expected vs mean paths
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
ES = S0 * np.exp(mu * t)   # expected GBM path

plt.plot(t, ES, 'k', label="Expected path")
plt.plot(t, S.mean(axis=1), ':k', label="Mean path")
plt.plot(t, S[:, ::1000], linewidth=0.5, alpha=0.5)
plt.xlabel("t")
plt.ylabel("S")
plt.title("Geometric Brownian Motion under Heston volatility")
plt.legend()
plt.tight_layout()
plt.savefig("hestonpathsS.png", dpi=150)
plt.show()
plt.close()

# %%
# ------------------------------------------------------------
# PDF histograms of S_t at t = 0.1, 0.4, 1.0
# ------------------------------------------------------------

plt.figure(figsize=(7, 12))

# time indices
idx1 = int(0.1 / dt)
idx2 = int(0.4 / dt)
idx3 = -1

bins = np.arange(0, 3.5 + 0.035, 0.035)

plt.subplot(3,1,1)
plt.hist(S[idx1, :], bins=bins, density=True)
plt.ylabel("f_S(x, 0.1)")
plt.xlim(0, 3.5)
plt.ylim(0, 3.5)
plt.title("Heston model: PDF of S at different times")

plt.subplot(3,1,2)
plt.hist(S[idx2, :], bins=bins, density=True)
plt.ylabel("f_S(x, 0.4)")
plt.xlim(0, 3.5)
plt.ylim(0, 3.5)

plt.subplot(3,1,3)
plt.hist(S[idx3, :], bins=bins, density=True)
plt.xlabel("x")
plt.ylabel("f_S(x, 1.0)")
plt.xlim(0, 3.5)
plt.ylim(0, 3.5)

plt.tight_layout()
plt.savefig("hestondensityS.png", dpi=150)
plt.show()

# %%
# ------------------------------------------------------------
# Probability density function of the log price at different times
# via FFT inversion of the characteristic function
# ------------------------------------------------------------

# -------------------------------------------------------------------------
# % Conditional characteristic function of the log price for the Heston SV model
# % Cui, del Bano Rollin, Germano, European Journal of Operational Research 263, 625, 2017
# % function phi = cf (xi, tau,r, q, kappa, vbar, sigma, rho, v0, x0)
# % Equivalent expressions of the conditional characteristic function
# % Schoutens, Simons and Tistaert (2004), Eq. (18)
# % Cui, del Bano Rollin and Germano (2017), Eqs. (11)-(13)
# % c = kappa - sigma*rho*1i*xi;
# % d = sqrt(c.^2 + sigma^2*(xi.^2 + 1i*xi));
# % g = (c - d) ./ (c + d);
# % beta = kappa*vbar/sigma^2;
# % phi = exp(1i*xi*(r-q)*tau + beta*((c-d)*tau + 2*log((1-g)./(1-g.*exp(-d*tau)))) ...
# %      + 1i*xi*x0 + v0/sigma^2*(c-d).*(1-exp(-d*tau))./(1-g.*exp(-d*tau)));
# %
# % Alternative expression (Cui et al. 2017), Eqs. (15)-(18)
# % A1 = (xi.^2+1i*xi).*sinh(d*tau/2);
# % A2 = d/v0.*cosh(d*tau/2) + c/v0.*sinh(d*tau/2);
# % D  = log(d/v0) + (kappa-d)*tau/2 - log((d+c)/(2*v0) + (d-c)/(2*v0).*exp(-d*tau));
# % phi = exp(1i*xi*(x0+(r-q)*tau) - kappa*vbar*rho*tau*1i*xi/sigma - A1./A2 + 2*beta*D);
# -------------------------------------------------------------------------
def cf(xi, tau, r, q, kappa, vbar, sigma, rho, v0, x0):
    c = kappa - sigma * rho * 1j * xi
    d = np.sqrt(c**2 + sigma**2 * (xi**2 + 1j * xi))
    g = (c - d) / (c + d)
    beta = kappa * vbar / sigma**2
    return np.exp(
        1j * xi * (r - q) * tau
        + beta * ((c - d) * tau + 2 * np.log((1 - g) / (1 - g * np.exp(-d * tau))))
        + 1j * xi * x0
        + v0 / sigma**2 * (c - d) * (1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau))
    )

# Grids
ngrid = 2048
width = 50.0
N = ngrid // 2
dx = width / ngrid
x = dx * (np.arange(-N, N))
dxi = 2 * np.pi / width
xi = dxi * (np.arange(-N, N))

# subplots

for i, e in enumerate([0.1, 0.4, 1.0]):

    datapoint = int(e * nsteps)
    plt.subplot(3,1,i + 1)
    plt.hist(X[datapoint, :], bins=x, density=True, edgecolor='black')
    phi = cf(xi, datapoint/nsteps, r, 0.0, kappa, Vbar, sigmaV, rho, V0, 0.0)
    f = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(phi)))) * dxi / (2*np.pi)
    plt.plot(x, f, 'r', linewidth=1)
    plt.ylabel(f"$f_X(x,{e})$")
    plt.xlim([-1, 1])
    plt.ylim([0, 4.5])

    if i == 0:
        plt.legend(["Monte Carlo", "Fourier"])
        plt.title("Heston model: PDF of the log price at t = 0.1, 0.4, 1.0")

plt.tight_layout()
plt.savefig("hestondensityX.png", dpi=150)
plt.show()
plt.close()