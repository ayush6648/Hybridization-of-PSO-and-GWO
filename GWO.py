import numpy as np


class GWO:
    """Grey Wolf Optimizer (GWO).

    Implements the standard GWO as described in:
    Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey Wolf Optimizer.
    Advances in Engineering Software, 69, 46-61.
    """

    def __init__(self, fit_func, num_dim=30, num_particle=20, max_iter=100,
                 lb=-100, ub=100):
        self.fit_func = fit_func
        self.num_dim = num_dim
        self.num_particle = num_particle
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub

        self.score_alpha = np.inf
        self.score_beta = np.inf
        self.score_delta = np.inf
        self.X_alpha = np.zeros(self.num_dim)
        self.X_beta = np.zeros(self.num_dim)
        self.X_delta = np.zeros(self.num_dim)

        self.result = np.zeros(self.max_iter)

        # Interaction coefficients
        self.C1 = 0.5
        self.C2 = 0.5
        self.C3 = 0.5

        self.X = np.random.uniform(
            low=self.lb, high=self.ub,
            size=(self.num_particle, self.num_dim)
        )

        self._iter = 0

    def opt(self):
        while self._iter < self.max_iter:
            for i in range(self.num_particle):
                score = self.fit_func(self.X[i, :])

                if score < self.score_alpha:
                    self.score_alpha = score
                    self.X_alpha = self.X[i, :].copy()
                elif score < self.score_beta:
                    self.score_beta = score
                    self.X_beta = self.X[i, :].copy()
                elif score < self.score_delta:
                    self.score_delta = score
                    self.X_delta = self.X[i, :].copy()

            # Linearly decrease a from 2 to 0
            a = 2 - 2 * self._iter / self.max_iter

            for i in range(self.num_particle):
                r1 = np.random.uniform(size=self.num_dim)
                A1 = 2 * a * r1 - a
                D_alpha = np.abs(self.C1 * self.X_alpha - self.X[i, :])
                X1 = self.X_alpha - A1 * D_alpha

                r1 = np.random.uniform(size=self.num_dim)
                A2 = 2 * a * r1 - a
                D_beta = np.abs(self.C2 * self.X_beta - self.X[i, :])
                X2 = self.X_beta - A2 * D_beta

                r1 = np.random.uniform(size=self.num_dim)
                A3 = 2 * a * r1 - a
                D_delta = np.abs(self.C3 * self.X_delta - self.X[i, :])
                X3 = self.X_delta - A3 * D_delta

                self.X[i, :] = np.clip((X1 + X2 + X3) / 3, self.lb, self.ub)

            self._iter += 1
            self.result[self._iter - 1] = self.score_alpha

    def return_result(self):
        return self.result
