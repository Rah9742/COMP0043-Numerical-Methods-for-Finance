import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

def simulate_heston(npaths, T, nsteps, r, kappa, theta, xi, rho, S0, V0, rng=None):
    """
    Returns (t, V, X, S) for the Heston model.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = T / nsteps
    t = np.linspace(0.0, T, nsteps + 1)

    V = np.empty((nsteps + 1, npaths), dtype=float)
    X = np.empty((nsteps + 1, npaths), dtype=float)

    V[0, :] = V0
    X[0, :] = np.log(S0)

    Z1 = rng.standard_normal(size=(nsteps, npaths))
    Z2 = rng.standard_normal(size=(nsteps, npaths))
    Z_v = Z1
    Z_s = rho * Z1 + np.sqrt(1.0 - rho**2) * Z2

    for i in range(nsteps):
        v_prev = V[i, :]
        v_pos = np.maximum(v_prev, 0.0)

        dW_v = np.sqrt(dt) * Z_v[i, :]
        dW_s = np.sqrt(dt) * Z_s[i, :]

        V[i + 1, :] = v_prev + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * dW_v
        V[i + 1, :] = np.maximum(V[i + 1, :], 0.0)

        X[i + 1, :] = X[i, :] + (r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dW_s

    S = np.exp(X)
    return t, V, X, S

# Example simulation call (this can be changed or commented out)
t, V, X, S = simulate_heston(
    npaths=5000, T=1.0, nsteps=200,
    r=0.03, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7,
    S0=100.0, V0=0.04
)

# ============================================================
# CIR / Feller utilities reused from your FSRP script
# dV = kappa (theta - V) dt + xi sqrt(V) dW^v
# ============================================================

def cir_mean(t, V0, theta, kappa):
    """E[V_t] for CIR."""
    t = np.asarray(t, dtype=float)
    return theta + (V0 - theta) * np.exp(-kappa * t)

def cir_long_run_std(xi, theta, kappa):
    """Long run standard deviation of CIR."""
    return xi * np.sqrt(theta / (2.0 * kappa))

def cir_feller_ratio(kappa, theta, xi):
    """Feller ratio 2*kappa*theta/xi^2."""
    return 2.0 * kappa * theta / (xi ** 2)

# If you really want to, you can also bring in your cir_em_moment_coeffs
# and cir_em_step here, but for Heston it is more transparent to use
# a simple full truncation Euler scheme for V so that correlation is
# exactly tied to the Brownian increments used for S.


# ============================================================
# Correlated normal generator
# ============================================================

def generate_correlated_normals(rho, nsteps, npaths, rng=None):
    """
    Generate two fields of standard normals (Z_v, Z_s) of shape (nsteps, npaths)
    with Corr(Z_v, Z_s) = rho.
    """
    if rng is None:
        rng = np.random.default_rng()
    Z1 = rng.standard_normal(size=(nsteps, npaths))
    Z2 = rng.standard_normal(size=(nsteps, npaths))
    Z_v = Z1
    Z_s = rho * Z1 + np.sqrt(1.0 - rho**2) * Z2
    return Z_v, Z_s


# ============================================================
# Diagnostics and plots
# ============================================================

# %% Variance: sample vs analytic CIR moments

# Heston parameters (risk neutral)
r = 0.03              # risk free rate
kappa = 2.0           # mean reversion speed of variance
theta = 0.04          # long run variance
xi = 0.5              # vol of vol
rho = -0.7            # correlation between price and variance

S0 = 100.0            # initial price
V0 = 0.04             # initial variance (often set to theta)

EX_V = cir_mean(t, V0, theta, kappa)
sdev_infty = cir_long_run_std(xi, theta, kappa)

plt.figure(figsize=(8, 5))
step = max(5000 // 25, 10)
plt.plot(t, V[:, ::step], linewidth=0.5, alpha=0.6)
plt.plot(t, EX_V, 'r', label='Analytic mean of V')
plt.plot(t, V.mean(axis=1), ':k', label='Sample mean of V')
plt.plot(t, theta * np.ones_like(t), 'k--', label=r'$\theta$')
plt.xlabel('t')
plt.ylabel('V')
plt.ylim([0.0, theta + 4 * sdev_infty])
plt.legend(loc='best')
plt.title('Heston variance process paths and mean')
plt.tight_layout()
plt.show()

# %% Price paths
plt.figure(figsize=(8, 5))
plt.plot(t, S[:, ::step], linewidth=0.5, alpha=0.6)
plt.xlabel('t')
plt.ylabel('S')
plt.title('Heston model price paths')
plt.tight_layout()
plt.show()

# %% Distribution of S_T and V_T
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(S[-1, :], bins=50, density=True, alpha=0.7)
plt.xlabel(r'$S_T$')
plt.ylabel('Density')
plt.title('Terminal price distribution under Heston')

plt.subplot(1, 2, 2)
plt.hist(V[-1, :], bins=50, density=True, alpha=0.7)
plt.xlabel(r'$V_T$')
plt.ylabel('Density')
plt.title('Terminal variance distribution under Heston')

plt.tight_layout()
plt.show()

# %% Heston PDF of log price at different times t = 0.1, 0.4, 1.0


times = [0.1, 0.4, 1.0]
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# --------------------------------------------------------------
# Heston characteristic function and Fourier inversion for PDF
# --------------------------------------------------------------

def heston_cf(u, t, r, kappa, theta, xi, rho, V0, X0):
    """
    Characteristic function φ(u) = E[e^{iu X_t}] of the log price X_t.
    Risk neutral version.
    """
    a = kappa * theta
    b = kappa - rho * xi * 1j * u
    d = np.sqrt(b*b + (xi*xi) * (u*u + 1j*u))
    g = (b - d) / (b + d)

    exp_dt = np.exp(-d * t)
    C = r * 1j * u * t + a/ (xi*xi) * ((b - d) * t - 2.0 * np.log((1 - g*exp_dt)/(1 - g)))
    D = (b - d) / (xi*xi) * ((1 - exp_dt)/(1 - g*exp_dt))

    return np.exp(C + D * V0 + 1j * u * X0)

def heston_pdf_fourier(xgrid, t, r, kappa, theta, xi, rho, V0, X0):
    """
    Compute PDF via Fourier inversion of (φ(u) e^{-iux0}) trick.
    """
    # u grid for FFT-type inversion
    umax = 200
    du = 0.25
    u = np.arange(0, umax, du)

    phi = heston_cf(u, t, r, kappa, theta, xi, rho, V0, X0)

    pdf = np.zeros_like(xgrid, float)
    for j, x in enumerate(xgrid):
        integrand = np.real(np.exp(-1j * u * x) * phi)
        pdf[j] = du/np.pi * np.trapezoid(integrand, u)

    # normalise to integrate to 1
    area = np.trapezoid(pdf, xgrid)
    if area > 0:
        pdf = pdf / area

    return pdf


def heston_cf_logreturn(u, tau, r, kappa, theta, xi, rho, v0):
    """
    Characteristic function φ(u) = E[exp(i u X_T)] of the Heston log-return X_T.
    Here X_T = log(S_T / S_0), so X_0 = 0.
    Parameters follow the lecture notation:
        kappa: mean reversion of variance
        theta: long run variance
        xi   : vol of vol (sigma in the notes)
        rho  : corr(dW^S, dW^V)
        v0   : initial variance
        r    : risk free rate
        tau  : maturity T
    """
    u = np.asarray(u, dtype=complex)

    a = kappa * theta
    b = kappa - rho * xi * 1j * u
    d = np.sqrt(b * b + (xi * xi) * (u * u + 1j * u))
    g = (b - d) / (b + d)

    exp_dt = np.exp(-d * tau)
    C = 1j * u * r * tau + (a / (xi * xi)) * (
        (b - d) * tau - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    D = (b - d) / (xi * xi) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))

    # X_0 = 0, so there is no extra i u X0 term
    return np.exp(C + D * v0)

def heston_pdf_fft(N, tau, r, kappa, theta, xi, rho, v0):
    """
    FFT-based inversion as in the lecture notes.
    Steps:
      1) u-grid: u_j = j - N/2, j = 0,...,N-1
      2) x-grid: x_j = (j - N/2) / N * (2π)
      3) compute cf on u-grid
      4) multiply by alternating sign h_j = (-1)^j
      5) FFT, switch sign pattern back, divide by 2π
    Returns (x_grid, pdf_fft).
    """
    N = int(N)
    j = np.arange(N)
    u = j - N / 2.0

    cf_vals = heston_cf_logreturn(u, tau, r, kappa, theta, xi, rho, v0)

    h = (-1.0) ** j
    f = h * cf_vals

    tmpp = np.fft.fft(f, n=N)
    pdf = np.maximum(np.real(tmpp * h / (2.0 * np.pi)), 0.0)

    x_grid = (j - N / 2.0) / N * (2.0 * np.pi)
    return x_grid, pdf


# --------------------------------------------------------------
# Plot MC histogram + Fourier PDF for t = 0.1, 0.4, 1.0
# --------------------------------------------------------------

for ax, tt in zip(axes, times):
    # Select nearest index in simulated array
    idx = np.argmin(np.abs(t - tt))
    Xtt = X[idx, :] - np.log(S0)

    # Histogram
    ax.hist(Xtt, bins=40, density=True, alpha=0.6, label="Monte Carlo")

    # Fourier inversion density
    xmin = Xtt.min()
    xmax = Xtt.max()
    xpad = 0.15 * (xmax - xmin)

    xgrid = np.linspace(xmin - xpad, xmax + xpad, 600)

    pdf = heston_pdf_fourier(xgrid, tt, r, kappa, theta, xi, rho, V0, X0=0)
    ax.plot(xgrid, pdf, "r", label="Fourier")

    # === FFT inversion (overlay) ===
    N_fft = 512
    x_fft, pdf_fft = heston_pdf_fft(
        N_fft,
        tt,
        r,
        kappa,
        theta,
        xi,
        rho,
        V0
    )

    # Shift FFT x-grid to align with histogram domain
    mask = (x_fft >= xmin - xpad) & (x_fft <= xmax + xpad)
    ax.plot(x_fft[mask], pdf_fft[mask], "--g", label="FFT")

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_title(f"PDF of log return ΔX at t = {tt}")
    ax.set_xlabel("x")
    ax.set_ylabel(f"$f_X (x,{tt})$")

axes[0].legend()
plt.tight_layout()
plt.show()
