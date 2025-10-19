import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Parameters
# ============================================
ngrid = 200       # grid resolution
nsample = 10_000  # number of random samples

# ============================================
# (a) Theoretical distributions
# ============================================

# Define support for U and X
u_edges = np.linspace(-np.pi/2, np.pi/2, ngrid + 1)
x_edges = np.linspace(0, 1, ngrid + 1)

# Bin centres (for plotting)
u_centres = 0.5 * (u_edges[:-1] + u_edges[1:])
x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])

# Theoretical PDFs
f_u_theory = np.full_like(u_centres, 1 / np.pi, dtype=float)  # U ~ Uniform(-pi/2, pi/2)
f_x_theory = 2 / (np.pi * np.sqrt(1 - np.clip(x_centres, 0, 1 - 1e-12)**2))  # X = cos(U)

# ============================================
# (b) Monte Carlo sampling
# ============================================

# Draw samples
u_samples = np.random.uniform(-np.pi/2, np.pi/2, nsample)
x_samples = np.cos(u_samples)

# Estimate PDFs using histograms (density=True normalises area = 1)
f_u_sampled, _ = np.histogram(u_samples, bins=u_edges, density=True)
f_x_sampled, _ = np.histogram(x_samples, bins=x_edges, density=True)

# ============================================
# (c) Plot results
# ============================================

# Helper function for consistent plot formatting
def plot_pdf(x, f_sampled, f_theory, title, xlabel='x', ylabel='f(x)'):
    plt.plot(x, f_sampled, 'r', label='Sampled')
    plt.plot(x, f_theory, 'b', label='Theory')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

plt.figure(figsize=(8, 5))
plot_pdf(u_centres, f_u_sampled, f_u_theory, 'PDF of U ~ Uniform(-π/2, π/2)', xlabel='U', ylabel='f_U(U)')
plt.ylim(0, 1)

plt.figure(figsize=(8, 5))
plot_pdf(x_centres, f_x_sampled, f_x_theory, 'PDF of X = cos(U)', xlabel='X', ylabel='f_X(X)')

plt.tight_layout()
plt.show()