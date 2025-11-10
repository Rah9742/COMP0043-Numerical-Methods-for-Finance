import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

# %% ===============================
# 1. Fokker–Planck (Drift–Diffusion Equation)
# ===============================
mu = -0.05  # drift
D = 0.5     # diffusion

x = np.arange(-1, 1.02, 0.02)
t = np.arange(0.1, 1.025, 0.025)
x, t = np.meshgrid(x, t)

f = 1 / (2 * np.sqrt(np.pi * D * t)) * np.exp(-(x - mu * t)**2 / (4 * D * t))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, t, f, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('f')
ax.set_title('Solution of the Fokker–Planck equation for arithmetic Brownian motion\n'
             'with μ = −0.05, σ = 0.4')
ax.view_init(24, 30)
plt.tight_layout()
# plt.savefig("abmfpe.png", dpi=300)
plt.show()
plt.close()

# %% ===============================
# 2. Black–Scholes–Merton Equation (S,t)
# ===============================
T = 1.0
K = 1.1
r = 0.05
q = 0.02
sigma = 0.4

S = np.arange(0, 2.05, 0.05)
t = np.arange(0, T + 0.025, 0.025)
S, t = np.meshgrid(S, t)

# Avoid division by zero
tau = T - t
tau[tau == 0] = 1e-12

d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
d2 = (np.log(S / K) + (r - q - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

# Call
Vc = S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
Vc[-1, :] = np.maximum(S[-1, :] - K, 0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S, t, Vc, cmap='plasma')
ax.set_xlabel('S')
ax.set_ylabel('t')
ax.set_zlabel('V_c')
ax.set_title('Call')
ax.view_init(24, -30)
plt.tight_layout()
# plt.savefig("bsc.png", dpi=300)
plt.show()
plt.close()

# Put
Vp = K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)
Vp[-1, :] = np.maximum(K - S[-1, :], 0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S, t, Vp, cmap='inferno')
ax.set_xlabel('S')
ax.set_ylabel('t')
ax.set_zlabel('V_p')
ax.set_title('Put')
ax.view_init(24, 30)
plt.tight_layout()
# plt.savefig("bsp.png", dpi=300)
plt.show()
plt.close()

# %% ===============================
# 3. Black–Scholes Equation as function of log price (x_a)
# ===============================
S0 = 1.0  # initial price (assumed)
k = np.log(K / S0)

xa = np.arange(-1, 1.05, 0.05)
t = np.arange(0, T + 0.025, 0.025)
xa, t = np.meshgrid(xa, t)

tau = T - t
tau[tau == 0] = 1e-12

d1 = (xa - k + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
d2 = (xa - k + (r - q - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

# Call
Vc = S0 * (np.exp(xa - q * tau) * norm.cdf(d1)
           - np.exp(k - r * tau) * norm.cdf(d2))
Vc[-1, :] = S0 * np.maximum(np.exp(xa[-1, :]) - np.exp(k), 0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xa, t, Vc, cmap='plasma')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('V_c')
ax.set_title('Call (log-price)')
ax.view_init(24, -30)
plt.tight_layout()
# plt.savefig("bscx.png", dpi=300)
plt.show()
plt.close()

# Put
Vp = S0 * (np.exp(k - r * tau) * norm.cdf(-d2)
           - np.exp(xa - q * tau) * norm.cdf(-d1))
Vp[-1, :] = S0 * np.maximum(np.exp(k) - np.exp(xa[-1, :]), 0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xa, t, Vp, cmap='inferno')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('V_p')
ax.set_title('Put (log-price)')
ax.view_init(24, 30)
plt.tight_layout()
# plt.savefig("bspx.png", dpi=300)
plt.show()
plt.close()