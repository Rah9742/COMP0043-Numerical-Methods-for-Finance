import numpy as np
import matplotlib.pyplot as plt

# %% Mean Reversion
expiry = 10
timestep = np.linspace(0, expiry, 100)

mu = 100
alpha = 1

# Mean reversion function: X(t) = mu + (x0 - mu) * exp(-alpha * t)
def sol(x0, alpha, timestep):
    return mu + (x0 - mu) * np.exp(-alpha * timestep)

# %% Plot solutions
plt.figure(figsize=(8, 5), facecolor='w')

plt.plot(timestep, sol(100, alpha, timestep), 'k', label=r'Model 1: $x_0=100$, $\mu=100$, $\alpha=1$')
plt.plot(timestep, sol(120, alpha, timestep), 'g', label=r'Model 2: $x_0=120$, $\mu=100$, $\alpha=1$')
plt.plot(timestep, sol(80, alpha, timestep), 'b', label=r'Model 3: $x_0=80$, $\mu=100$, $\alpha=1$')
plt.plot(timestep, sol(120, 2 * alpha, timestep), 'c', label=r'Model 4: $x_0=120$, $\mu=100$, $\alpha=2$')
plt.plot(timestep, sol(80, 2 * alpha, timestep), 'r', label=r'Model 5: $x_0=80$, $\mu=100$, $\alpha=2$')

plt.xlabel('Time (years)')
plt.ylabel('X(t)')
plt.title('Convergence of X(t) towards its long-run value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('LecBMFigMeanReversion.png', dpi=300)
plt.show()