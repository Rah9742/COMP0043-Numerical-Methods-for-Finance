import numpy as np
import matplotlib.pyplot as plt

def lcg_int(seed, a, b, M, n):
    """Linear Congruential Generator returning n INTEGER states in [0, M-1]."""
    x = seed
    out = []
    for _ in range(n):
        x = (a * x + b) % M
        out.append(x)
    return out

def fibonacci_rng(a, b, u, v, M, seed, nsamples, op="minus"):
    """
    Lagged-Fibonacci generator:
      X_n = (X_{n-u} ± X_{n-v}) mod M
    Returns nsamples uniforms U_n = X_n / M.
    """
    if not (1 <= u < v):
        raise ValueError("Require 1 <= u < v.")

    # 1) Get integer seeds X_0..X_{v-1} via LCG
    X = lcg_int(seed, a, b, M, v)

    # 2) Generate X_v .. X_{v+nsamples-1}
    for n in range(v, v + nsamples):
        if op == "minus":
            xn = (X[n - u] - X[n - v]) % M
        elif op == "plus":
            xn = (X[n - u] + X[n - v]) % M
        else:
            raise ValueError("op must be 'minus' or 'plus'.")
        X.append(xn)

    # 3) Normalise the last nsamples values only
    X_tail = np.array(X[v:v + nsamples], dtype=float)
    return X_tail / M  # U_n in [0,1)

def scatter_square(a, b, u, v, M, seed, nsamples=4000, op="minus"):
    U = fibonacci_rng(a, b, u, v, M, seed, nsamples, op=op)

    # Build (U_n, U_{n-1}) pairs
    x = U[1:]           # U_n
    y = U[:-1]          # U_{n-1}

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'rx', markersize=2)
    plt.xlabel('U\u2099')                 # Uₙ
    plt.ylabel('U\u2099\u208B\u2081')     # Uₙ₋₁
    plt.title(f'Unit Square Scatter of Lagged Fibonacci '
              f'(nsamples={nsamples}, u={u}, v={v}, M={M}, op={op})')
    plt.tight_layout()
    plt.show()

# Example (tweak to match Seydel’s exact parameters)
scatter_square(a=1597, b=51749, u=5, v=17, M=644_980, seed=24_234, nsamples=4000, op="minus")