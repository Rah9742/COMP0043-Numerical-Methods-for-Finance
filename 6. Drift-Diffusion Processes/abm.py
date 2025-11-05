import numpy as np
import matplotlib.pyplot as plt


def get_parameters():
    npaths = 50
    T = 1
    nsteps = 200
    dt = T/nsteps
    t = np.arange(0, T, dt)
    mu, sigma = -0.05, 0.4

    return npaths, T, nsteps, dt, t, mu, sigma


def compute_sde(mu, dt, sigma, npaths, nsteps, t):
    dX = mu*dt + np.multiply(sigma*np.sqrt(dt), np.random.randn(npaths, nsteps))
    dX = np.insert(dX, 0, 0, axis=1)
    dX = np.delete(dX, -1, axis=1)
    X = np.cumsum(dX, axis=1)
    EX = mu*t
    return X, EX


def run_and_plot():
    npaths, T, nsteps, dt, t, mu, sigma = get_parameters()
    X, EX = compute_sde(mu, dt, sigma, npaths, nsteps, t)

    plt.figure(1)
    plt.plot(t, EX, linestyle='-.', color="b", marker='o', label='Expected path')
    print(X)
    plt.plot(t, X[0:-1, :].T)
    plt.xlabel('x')
    plt.title('Paths of an arithmetic Brownian motion dX(t) = mu*dt + sigma*dW(t)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_and_plot()
