import numpy as np

class Clusterer:
    def __init__(self, K: int, D: int, sigma: float, lambda_: float, learning_rate: float):
        self.K = K
        self.D = D
        self.mu = np.random.rand(K, D)
        self.sigma = sigma
        self.lambda_ = lambda_
        self.learning_rate = learning_rate

        self.i: list[int] = []
        self.j: list[int] = []
        for i in range(0, K):
            for j in range(0, K):
                if i != j:
                    self.i.append(i)
                    self.j.append(j)

        #Parameters for rewards calculations for the centroids
        self.cluster_rewards = np.zeros(K)


    def f(self, x: np.ndarray, i: None | int | list[int] = None) -> np.ndarray:
            if i == None:
                return np.exp(-np.square(x - self.mu).sum(axis=1) / self.sigma)
            if isinstance(i, int):
                return np.exp(-np.square(x - self.mu[i]).sum() / self.sigma)
            return np.exp(-np.square(x - self.mu[i]).sum(axis=1) / self.sigma)

    def update(self, x: np.ndarray):
        weighting_function = x+1
        #Update centroids
        diff = self.f(x).reshape(-1, 1) * (x - self.mu) - (2.0 * self.lambda_ *
                                                        (self.mu[self.j] - self.mu[self.i]) * self.f(self.mu[self.j], self.i).reshape(-1, 1)).reshape(
                                                            -1, self.K - 1, self.mu.shape[1]).sum(axis=1)

        self.mu += self.learning_rate * diff / self.sigma

        #Update rewards

    

    def f_min(self) -> float:
        # sqrt(a/2) is the inflection point and equal to solution of d^2/dx^2 e^(-x^2/a) = 0 given x>0, a>0
        return np.exp(-np.square(3. * np.sqrt(self.sigma / 2.) - 0.).sum() / self.sigma)
