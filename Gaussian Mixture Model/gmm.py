import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def generate_data(d=2, plot=True):
    num_samples = 2000
    s = 4
    means = [np.array([0, 0]), np.array([0, s]), np.array([s, s])]
    X = np.zeros((num_samples, d))
    X[:1200, :] = np.random.randn(1200, d) * 2 + means[0]
    X[1200:1800, :] = np.random.randn(600, d) + means[1]
    X[1800:, :] = np.random.randn(200, d) * 0.5 + means[2]
    if plot:
        plt.scatter(X[:, 0], X[:, 1])
        plt.show()
    return X


def gmm(X, K, max_iter=20, smoothing=1e-2):
    N, D = X.shape
    M = np.zeros((K, D))
    R = np.zeros((N, K))
    C = np.zeros((K, D, D))
    pi = np.ones(K) / K

    for k in range(K):
        M[k] = X[np.random.choice(N)]
        C[k] = np.eye(D)

    lls = []
    weighted_pdfs = np.zeros((N, K))
    for i in range(max_iter):
        for k in range(K):
            weighted_pdfs[:, k] = pi[k] * multivariate_normal.pdf(X, M[k], C[k])
        R = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)

        for k in range(K):
            Nk = R[:, k].sum()
            pi[k] = Nk / N
            M[k] = R[:, k].dot(X) / Nk
            delta = X - M[k]  # N x D
            R_delta = np.expand_dims(R[:, k], -1) * delta
            C[k] = R_delta.T.dot(delta) / Nk + np.eye(D) * smoothing

        ll = np.log(weighted_pdfs.sum(axis=1)).sum()
        lls.append(ll)
        if i > 0:
            if np.abs(lls[i] - lls[i - 1]) < 0.1:
                break

    plt.plot(lls)
    plt.title("Log-Likelihood")
    plt.show()

    random_colors = np.random.random((K, 3))
    colors = R.dot(random_colors)
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.show()

    print("pi:", pi)
    print("means:", M)
    print("covariances:", C)
    return R


def main():
    X = generate_data()
    gmm(X, 3)


if __name__ == '__main__':
    main()
