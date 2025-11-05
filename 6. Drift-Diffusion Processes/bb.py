import numpy as np
import matplotlib.pyplot as plt


def get_parameters():
    npaths = 50
    T = 1
    nsteps = 200
    dt = T/nsteps
    t = np.arange(0, T, dt)
    sigma = 0.3
    a = 0.8
    b = 1

    return npaths, T, nsteps, dt, t, sigma, a, b


def compute_sde(a, b, dt, sigma, npaths, nsteps, T, t):
    dX = np.concatenate((a*np.ones((1, npaths)), np.zeros((nsteps-1, npaths)), b*np.ones((1, npaths))), axis=0)
    for i in range(0, nsteps):
        dX[i+1, :] = dX[i, :] + (b - dX[i, :])/(nsteps - i + 1) + sigma*np.random.randn(1, npaths)*np.sqrt(dt)
    EX = a + (b-a)/T*t
    return dX, EX


def run_and_plot():
    npaths, T, nsteps, dt, t, sigma, a, b = get_parameters()
    X, EX = compute_sde(a, b, dt, sigma, npaths, nsteps, T, t)

    plt.figure(1)
    plt.plot(t, EX, linestyle='-.', color="b", marker='o', label='Expected path')
    plt.plot(t, X[0:-1, :])
    plt.xlabel('x')
    plt.title('Paths of Brownian bridge dX = ((b-X)/(T-t))*dt + sigma*dW')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_and_plot()
