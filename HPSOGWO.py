import numpy as np


class HPSOGWO:
    """Hybrid Particle Swarm Optimization and Grey Wolf Optimizer (HPSOGWO).

    Implements the hybrid algorithm from:
    Singh, N., & Singh, S. B. (2017). Hybrid Algorithm of Particle Swarm
    Optimization and Grey Wolf Optimizer for Improving Convergence
    Performance. Journal of Applied Mathematics, 2017, 1-15.
    """

    def __init__(self, fitness, num_dim=30, pop_size=20, max_iter=100,
                 lb=-100, ub=100, a_max=2, a_min=0,
                 init_w=0.5, c1=0.5, c2=0.5, c3=0.5, k=0.2):
        self.fitness = fitness
        self.num_dim = num_dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.ub = ub
        self.lb = lb
        self.a_max = a_max
        self.a_min = a_min
        self.init_w = init_w
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.k = k
        self.v_max = self.k * (self.ub - self.lb)

        self.gbest_X = np.zeros(self.num_dim)
        self.gbest_F = np.inf
        self.result = np.zeros(self.max_iter)
        self.F_alpha = np.inf
        self.F_beta = np.inf
        self.F_delta = np.inf
        self.X_alpha = np.zeros(self.num_dim)
        self.X_beta = np.zeros(self.num_dim)
        self.X_delta = np.zeros(self.num_dim)

    def opt(self):
        self.X = np.random.uniform(
            low=self.lb, high=self.ub, size=(self.pop_size, self.num_dim)
        )
        self.V = np.random.uniform(
            low=self.lb, high=self.ub, size=(self.pop_size, self.num_dim)
        )

        for g in range(self.max_iter):
            F = np.array([self.fitness(self.X[i]) for i in range(self.pop_size)])

            best_idx = F.argmin()
            if F[best_idx] < self.gbest_F:
                self.gbest_F = F[best_idx]
                self.gbest_X = self.X[best_idx].copy()

            self.result[g] = self.gbest_F

            # Update wolf hierarchy
            for i in range(self.pop_size):
                if F[i] < self.F_alpha:
                    self.F_alpha = F[i]
                    self.X_alpha = self.X[i].copy()
                elif F[i] < self.F_beta:
                    self.F_beta = F[i]
                    self.X_beta = self.X[i].copy()
                elif F[i] < self.F_delta:
                    self.F_delta = F[i]
                    self.X_delta = self.X[i].copy()

            # Linearly decrease a and update inertia weight w
            a = self.a_max - (self.a_max - self.a_min) * (g / self.max_iter)
            self.w = self.init_w + np.random.uniform() / 2

            # Compute guided positions from alpha, beta, delta wolves
            r1 = np.random.uniform(size=(self.pop_size, self.num_dim))
            A = 2 * a * r1 - a
            D = np.abs(self.c1 * self.X_alpha - self.w * self.X)
            X1 = self.X_alpha - A * D

            r1 = np.random.uniform(size=(self.pop_size, self.num_dim))
            A = 2 * a * r1 - a
            D = np.abs(self.c2 * self.X_beta - self.w * self.X)
            X2 = self.X_beta - A * D

            r1 = np.random.uniform(size=(self.pop_size, self.num_dim))
            A = 2 * a * r1 - a
            D = np.abs(self.c3 * self.X_delta - self.w * self.X)
            X3 = self.X_delta - A * D

            # PSO-style velocity update guided by GWO positions
            r2 = np.random.uniform(size=(self.pop_size, self.num_dim))
            r3 = np.random.uniform(size=(self.pop_size, self.num_dim))
            r4 = np.random.uniform(size=(self.pop_size, self.num_dim))

            self.V = self.w * (
                self.V
                + self.c1 * r2 * (X1 - self.X)
                + self.c2 * r3 * (X2 - self.X)
                + self.c3 * r4 * (X3 - self.X)
            )
            self.V = np.clip(self.V, -self.v_max, self.v_max)

            self.X = np.clip(self.X + self.V, self.lb, self.ub)

    def return_result(self):
        return self.result
