import numpy as np
import matplotlib.pyplot as plt


def generate_data(d=2, k=5, plot=True):
    num_samples = 900
    s = 4
    means = [(np.random.random(d) * 10) for i in range(k)]
    X = np.zeros((num_samples, d))
    batch_size = num_samples // len(means)
    for i, val in enumerate(means):
        X[i * batch_size: (i + 1) * batch_size] = np.random.randn(batch_size, d) + val
    # np.random.shuffle(X)
    if plot:
        plt.scatter(X[:, 0], X[:, 1])
        plt.show()
    return X


def dist(u, v):
    diff = u - v
    return diff.dot(diff)


def get_cost(X, R, means):
    cost = 0
    for k in range(means.shape[0]):
        for n in range(X.shape[0]):
            cost += R[n, k] * dist(means[k], X[n])
    return cost
    # return sum([r[i, k] * dist(means[k], X[i]) for i in range(X.shape[0]) for k in range(means.shape[0])])


def k_means(X, K, beta=2.0, max_iter=30, plot=True):
    N, D = X.shape
    means = np.array([X[np.random.randint(X.shape[0])] for _ in range(K)])
    R = np.zeros((N, K))
    costs = []
    for i in range(max_iter):
        for k in range(K):
            for n in range(N):
                R[n, k] = np.exp(-beta * dist(means[k], X[n])) /\
                          np.sum([np.exp(-beta * dist(means[j], X[n])) for j in range(K)])

        for k in range(K):
            means[k] = R[:, k].dot(X) / R[:, k].sum()

        costs.append(get_cost(X, R, means))
        if i > 0:
            if abs(costs[-1] - costs[-2]) < 0.1:
                break

    print(f"COSTS: {costs}")
    if plot:
        plt.title("Cost per iteration")
        plt.plot(costs)
        plt.show()

        random_colors = np.random.random((K, 3))
        colors = R.dot(random_colors)
        plt.scatter(X[:, 0], X[:, 1], c=colors)
        plt.show()
    return means


def main():
    data = generate_data(d=2, k=5)
    print(f"Shape: {data.shape}")
    k_means(X=data, K=5)


if __name__ == '__main__':
    main()
