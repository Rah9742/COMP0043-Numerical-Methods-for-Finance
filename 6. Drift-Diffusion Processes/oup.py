import numpy as np
import matplotlib.pyplot as plt


def get_parameters():
    npaths = 50
    T = 1
    nsteps = 200
    dt = T/nsteps
    t = np.arange(0, T, dt)
    alpha, mu, sigma = 5, 0.07, 0.07

    return npaths, T, nsteps, dt, t, alpha,  mu, sigma


def compute_sde(alpha, mu, dt, sigma, npaths, nsteps, t):
    X_0 = 0.03
    dX = np.concatenate((X_0*np.ones((1, npaths)), np.zeros((nsteps, npaths))), axis=0)
    N = np.random.randn(nsteps, npaths)
    sdev = sigma*np.sqrt((1-np.exp(-2*alpha*dt))/(2*alpha))
    for i in range(0, nsteps-1):
        dX[i+1, :] = mu + (dX[i, :]-mu)*np.exp(-alpha*dt) + sdev*N[i, :]

    EX = mu + (X_0-mu)*np.exp(-alpha*t)
    return dX, EX


def run_and_plot():
    npaths, T, nsteps, dt, t, alpha, mu, sigma = get_parameters()
    X, EX = compute_sde(alpha, mu, dt, sigma, npaths, nsteps, t)

    plt.figure(1)
    plt.plot(t, EX, linestyle='-.', color="b", marker='o', label='Expected path')
    plt.plot(t, X[0:-1, :])
    plt.xlabel('x')
    plt.title('Paths of Ornstein-Uhlenbeck process dX = alpha*(mu-X)*dt + sigma*dW')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_and_plot()
