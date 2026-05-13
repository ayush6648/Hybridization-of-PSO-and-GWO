import numpy as np


class PSO:
    """Particle Swarm Optimization (PSO).

    Implements the standard PSO with inertia weight as described in:
    Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer.
    IEEE World Congress on Computational Intelligence, 69-73.
    """

    def __init__(self, fitness_func, c1=0.5, c2=0.5, dimension=30,
                 pop_size=20, max_iter=100, lb=-100, ub=100):
        self.dimension = dimension
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub

        self.c1 = c1    # cognitive constant (personal best)
        self.c2 = c2    # social constant (global best)
        self.w = 0.9    # inertia weight

        self.cost_func = fitness_func
        self.X = None       # positions
        self.V = None       # velocities
        self.p_best = None  # personal best positions
        self.p_best_cost = None  # personal best costs (cached to avoid recomputation)
        self.g_best = None  # global best position
        self.g_best_cost = np.inf

        self.evolution = []

    def random_init(self):
        self.X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dimension))
        self.V = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dimension))
        self.p_best = self.X.copy()
        self.p_best_cost = np.array([self.cost_func(x) for x in self.p_best])

        best_idx = np.argmin(self.p_best_cost)
        self.g_best = self.p_best[best_idx].copy()
        self.g_best_cost = self.p_best_cost[best_idx]

    def _update_bests(self):
        costs = np.array([self.cost_func(x) for x in self.X])
        improved = costs < self.p_best_cost
        self.p_best[improved] = self.X[improved].copy()
        self.p_best_cost[improved] = costs[improved]

        best_idx = np.argmin(self.p_best_cost)
        if self.p_best_cost[best_idx] < self.g_best_cost:
            self.g_best = self.p_best[best_idx].copy()
            self.g_best_cost = self.p_best_cost[best_idx]

    def _update_velocity(self):
        r1 = np.random.uniform(size=(self.pop_size, self.dimension))
        r2 = np.random.uniform(size=(self.pop_size, self.dimension))
        self.V = (
            self.w * self.V
            + self.c1 * r1 * (self.p_best - self.X)
            + self.c2 * r2 * (self.g_best - self.X)
        )

    def start(self):
        for _ in range(self.max_iter):
            self._update_bests()
            self._update_velocity()
            self.X = np.clip(self.X + self.V, self.lb, self.ub)
            self.evolution.append(self.g_best_cost)

    def return_result(self):
        return np.array(self.evolution)
