# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "matplotlib>=3.8",
# ]
# ///
#!/usr/bin/env python3

"""
Filename: COMP0043_Computer_Exam_Rahil_Shah.py
Author: Rahil Shah
Email: ucabr05@ucl.ac.uk
Date: 2025-12-03
Version: 1.0
Description:
    Solutions for COMP0043 Numerical Methods in Finance 2025 Computer Exam
    Numerical methods in:
    - Black-Scholes pricing
    - SDE discretisation
    - Monte Carlo
    - Binomial trees
    - Crank-Nicolson finite difference
"""


# %%
# Imports

import math
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import Generator, default_rng


# %%
# Matplotlib Global Style

plt.rcParams.update({
    # Font
    "font.size": 10,
    "font.family": "Times New Roman",

    # Figure sizing
    "figure.figsize": (6, 4),
    "figure.dpi": 120,

    # Line styles
    "lines.linewidth": 1.2,
    "lines.markersize": 4,

    # Axes
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,

    # Ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    # Legends
    "legend.frameon": False,
    "legend.fontsize": 10,

    # Saving
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# %%
# Utility helpers


def get_rng(seed: int = 42) -> Generator:
    """Return deterministic numpy random generator."""
    return default_rng(seed)


def plot_line(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Simple line plot helper for clarity in exam solutions."""
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %%
# Analytic Black-Scholes helpers


def _std_normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erf to avoid external dependencies."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t_maturity: float,
    is_call: bool = True,
) -> float:
    """Analytic Black-Scholes price for a European call or put option."""
    assert sigma >= 0.0
    assert t_maturity >= 0.0

    if t_maturity == 0.0 or sigma == 0.0:
        intrinsic = max(s0 - k, 0.0) if is_call else max(k - s0, 0.0)
        return intrinsic

    sqrt_t = math.sqrt(t_maturity)
    d1 = (
        math.log(s0 / k)
        + (r + 0.5 * sigma * sigma) * t_maturity
    ) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    if is_call:
        return s0 * _std_normal_cdf(d1) - k * math.exp(-r * t_maturity) * _std_normal_cdf(d2)

    return k * math.exp(-r * t_maturity) * _std_normal_cdf(-d2) - s0 * _std_normal_cdf(-d1)


# %%
# Root-finding


def newton_root(
    f,
    df,
    x0: float,
    atol: float = 1e-8,
    rtol: float = 1e-8,
    max_iter: int = 50,
) -> float:
    """Newton's method for f(x) = 0.

    Raises ValueError on lack of convergence.
    """
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0.0:
            raise ZeroDivisionError("Derivative became zero in Newton method.")

        x_new = x - fx / dfx
        if abs(x_new - x) < atol + rtol * abs(x):
            return x_new
        x = x_new

    raise ValueError("Newton method did not converge.")


# %%
# SDE discretisation schemes for GBM


def euler_maruyama(
    s0: float,
    t_maturity: float,
    n_steps: int,
    mu: float,
    sigma: float,
    rng: Generator,
) -> np.ndarray:
    """Euler-Maruyama discretisation of dS = mu*S*dt + sigma*S*dW.

    Returns the full path as a numpy array of shape (n_steps + 1,).
    """
    assert sigma >= 0.0
    assert n_steps > 0
    assert t_maturity > 0.0

    dt = t_maturity / n_steps
    sqrt_dt = math.sqrt(dt)

    s_path = np.zeros(n_steps + 1)
    s_path[0] = s0

    for i in range(n_steps):
        z = rng.normal()
        s = s_path[i]
        s_path[i + 1] = s + mu * s * dt + sigma * s * sqrt_dt * z

    return s_path


def milstein_scheme(
    s0: float,
    t_maturity: float,
    n_steps: int,
    mu: float,
    sigma: float,
    rng: Generator,
) -> np.ndarray:
    """Milstein discretisation of GBM dS = mu*S*dt + sigma*S*dW.

    Strong order 1.0 scheme, more accurate than Euler-Maruyama for GBM.
    """
    assert sigma >= 0.0
    assert n_steps > 0
    assert t_maturity > 0.0

    dt = t_maturity / n_steps
    sqrt_dt = math.sqrt(dt)

    s_path = np.zeros(n_steps + 1)
    s_path[0] = s0

    for i in range(n_steps):
        z = rng.normal()
        s = s_path[i]
        dW = sqrt_dt * z
        s_path[i + 1] = (
            s
            + mu * s * dt
            + sigma * s * dW
            + 0.5 * sigma * sigma * s * (dW * dW - dt)
        )

    return s_path


# %%
# Monte Carlo engines for European options on GBM


def simulate_gbm_paths(
    s0: float,
    t_maturity: float,
    n_steps: int,
    mu: float,
    sigma: float,
    n_paths: int,
    rng: Generator,
    use_milstein: bool = False,
) -> np.ndarray:
    """Simulate GBM paths using Euler-Maruyama or Milstein.

    Returns array of shape (n_paths, n_steps + 1).
    """
    assert n_paths > 0

    paths = np.zeros((n_paths, n_steps + 1))
    for i in range(n_paths):
        if use_milstein:
            paths[i] = milstein_scheme(
                s0=s0,
                t_maturity=t_maturity,
                n_steps=n_steps,
                mu=mu,
                sigma=sigma,
                rng=rng,
            )
        else:
            paths[i] = euler_maruyama(
                s0=s0,
                t_maturity=t_maturity,
                n_steps=n_steps,
                mu=mu,
                sigma=sigma,
                rng=rng,
            )

    return paths


def mc_price_european_gbm(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t_maturity: float,
    n_paths: int,
    n_steps: int,
    rng: Generator,
    is_call: bool = True,
    use_milstein: bool = False,
    antithetic: bool = False,
) -> tuple[float, float]:
    """Monte Carlo pricing of a European call/put on GBM.

    Returns (price, standard_error).
    """
    assert n_paths > 0

    if antithetic:
        # Use half paths and generate antithetic pairs
        base_paths = n_paths // 2
        payoffs = []
        for _ in range(base_paths):
            # Single-step analytic GBM for efficiency when antithetic
            z = rng.normal()
            s_t1 = s0 * math.exp(
                (r - 0.5 * sigma * sigma) * t_maturity
                + sigma * math.sqrt(t_maturity) * z
            )
            s_t2 = s0 * math.exp(
                (r - 0.5 * sigma * sigma) * t_maturity
                - sigma * math.sqrt(t_maturity) * z
            )
            if is_call:
                payoff_1 = max(s_t1 - k, 0.0)
                payoff_2 = max(s_t2 - k, 0.0)
            else:
                payoff_1 = max(k - s_t1, 0.0)
                payoff_2 = max(k - s_t2, 0.0)
            payoffs.append(0.5 * (payoff_1 + payoff_2))
        payoffs_arr = np.array(payoffs, dtype=float)
    else:
        paths = simulate_gbm_paths(
            s0=s0,
            t_maturity=t_maturity,
            n_steps=n_steps,
            mu=r,
            sigma=sigma,
            n_paths=n_paths,
            rng=rng,
            use_milstein=use_milstein,
        )
        s_terminal = paths[:, -1]
        if is_call:
            payoffs_arr = np.maximum(s_terminal - k, 0.0)
        else:
            payoffs_arr = np.maximum(k - s_terminal, 0.0)

    discount = math.exp(-r * t_maturity)
    discounted = discount * payoffs_arr
    price = float(np.mean(discounted))
    std_error = float(np.std(discounted, ddof=1) / math.sqrt(discounted.size))

    return price, std_error


# %%
# Binomial tree (CRR) for European options


def binomial_tree_european(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t_maturity: float,
    n_steps: int,
    is_call: bool = True,
) -> float:
    """CRR binomial tree price for European call or put."""
    assert n_steps > 0
    assert t_maturity >= 0.0

    if t_maturity == 0.0:
        intrinsic = max(s0 - k, 0.0) if is_call else max(k - s0, 0.0)
        return intrinsic

    dt = t_maturity / n_steps
    discount = math.exp(-r * dt)

    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)

    # Terminal payoffs
    values = np.zeros(n_steps + 1)
    for j in range(n_steps + 1):
        s_t = s0 * (u ** j) * (d ** (n_steps - j))
        if is_call:
            values[j] = max(s_t - k, 0.0)
        else:
            values[j] = max(k - s_t, 0.0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        values[: i + 1] = discount * (
            p * values[1 : i + 2] + (1.0 - p) * values[: i + 1]
        )

    return float(values[0])


# %%
# Finite difference: Crank-Nicolson for European options


def _solve_tridiagonal(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Thomas algorithm for tridiagonal system.

    Solves Ax = d where A has subdiag a, diag b, superdiag c.
    Arrays are modified in-place for efficiency.
    """
    n = d.size

    # Forward sweep
    for i in range(1, n):
        w = a[i] / b[i - 1]
        b[i] = b[i] - w * c[i - 1]
        d[i] = d[i] - w * d[i - 1]

    # Back substitution
    x = np.zeros_like(d)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


def crank_nicolson_european_call(
    s_max: float,
    k: float,
    r: float,
    sigma: float,
    t_maturity: float,
    n_s: int,
    n_t: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crank-Nicolson finite difference for a European call on non-dividend stock.

    Returns grid of option values V, spatial grid S, and time grid tau.
    """
    assert n_s > 1
    assert n_t > 0

    ds = s_max / n_s
    dt = t_maturity / n_t

    s_values = np.linspace(0.0, s_max, n_s + 1)
    t_values = np.linspace(0.0, t_maturity, n_t + 1)

    # Grid: rows = time levels, columns = space nodes
    v = np.zeros((n_t + 1, n_s + 1))

    # Terminal condition at maturity
    v[-1, :] = np.maximum(s_values - k, 0.0)

    # Boundary conditions in time
    # S=0 -> option worthless for call
    v[:, 0] = 0.0
    # S -> infinity, V ~ S - K*exp(-r(T-t))
    v[:, -1] = s_max - k * np.exp(-r * (t_maturity - t_values))

    # Coefficients for interior points
    j = np.arange(1, n_s)
    alpha = 0.25 * dt * (sigma * sigma * j * j - r * j)
    beta = -0.5 * dt * (sigma * sigma * j * j + r)
    gamma = 0.25 * dt * (sigma * sigma * j * j + r * j)

    # Matrices A and B are tridiagonal; we store as diagonals
    a = -alpha.copy()
    b = 1.0 - beta.copy()
    c = -gamma.copy()

    a_star = alpha.copy()
    b_star = 1.0 + beta.copy()
    c_star = gamma.copy()

    # Time stepping backwards
    for n in range(n_t - 1, -1, -1):
        # Right-hand side
        v_mid = v[n + 1, 1:-1]
        rhs = (
            a_star * v[n + 1, :-2]
            + b_star * v_mid
            + c_star * v[n + 1, 2:]
        )

        # Adjust for boundary terms
        rhs[0] -= a[0] * v[n, 0]
        rhs[-1] -= c[-1] * v[n, -1]

        # Solve tridiagonal system for interior nodes at time n
        v[n, 1:-1] = _solve_tridiagonal(
            a=a.copy(),
            b=b.copy(),
            c=c.copy(),
            d=rhs,
        )

    return v, s_values, t_values


# %%
# Template entry point for an exam question


def main_q1() -> None:
    """Entry point for Question 1.

    Adjust parameters and calls as needed for the specific question.
    """
    rng = get_rng(123)

    # Example parameters for demonstration only
    s0 = 100.0
    k = 100.0
    r = 0.05
    sigma = 0.2
    t_maturity = 1.0

    # Analytic price
    bs_call = black_scholes_price(
        s0=s0,
        k=k,
        r=r,
        sigma=sigma,
        t_maturity=t_maturity,
        is_call=True,
    )

    # Binomial price
    bt_call = binomial_tree_european(
        s0=s0,
        k=k,
        r=r,
        sigma=sigma,
        t_maturity=t_maturity,
        n_steps=200,
        is_call=True,
    )

    # Monte Carlo price
    price_mc, se_mc = mc_price_european_gbm(
        s0=s0,
        k=k,
        r=r,
        sigma=sigma,
        t_maturity=t_maturity,
        n_paths=50_000,
        n_steps=100,
        rng=rng,
        is_call=True,
        use_milstein=False,
        antithetic=True,
    )

    # Finite difference price: interpolate value at s0 from grid
    v_grid, s_grid, t_grid = crank_nicolson_european_call(
        s_max=4.0 * s0,
        k=k,
        r=r,
        sigma=sigma,
        t_maturity=t_maturity,
        n_s=200,
        n_t=200,
    )
    # At time 0, nearest node to s0
    idx_s0 = int(round(s0 / (s_grid[1] - s_grid[0])))
    fd_call = float(v_grid[0, idx_s0])

    print("Analytic BS call:", bs_call)
    print("Binomial call  :", bt_call)
    print("MC call        :", price_mc, "(SE =", se_mc, ")")
    print("FD CN call     :", fd_call)

    # Optional: plot FD surface slice at t=0
    plot_line(
        x=s_grid,
        y=v_grid[0, :],
        title="Crank-Nicolson European Call at t=0",
        xlabel="S",
        ylabel="V(0, S)",
    )


# %%
# Main guard: adjust to run specific exam question entry point


if __name__ == "__main__":
    main_q1()
