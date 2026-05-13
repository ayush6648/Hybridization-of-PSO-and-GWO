import numpy as np
import math


def _prod(it):
    p = 1
    for n in it:
        p *= n
    return p


def _ufun(x, a, k, m):
    """Penalty function used in F5 (Penalized Levy)."""
    return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < -a)


# ── Unimodal Benchmark Functions ──────────────────────────────────────────────

def F1(x):
    """Sphere function. Global min = 0."""
    return np.sum(x ** 2)


def F2(x):
    """Schwefel 2.22. Global min = 0."""
    return np.sum(np.abs(x)) + _prod(np.abs(x))


def F3(x):
    """Ackley function. Global min = 0."""
    dim = len(x)
    return (
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / dim))
        - np.exp(np.sum(np.cos(2 * math.pi * x)) / dim)
        + 20 + math.e
    )


def F4(x):
    """Griewank function. Global min = 0."""
    w = np.arange(1, len(x) + 1)
    return np.sum(x ** 2) / 4000 - _prod(np.cos(x / np.sqrt(w))) + 1


def F5(x):
    """Penalized Levy function. Global min = 0."""
    return (
        0.1 * (
            (np.sin(3 * math.pi * x[0])) ** 2
            + np.sum((x[:-1] - 1) ** 2 * (1 + (np.sin(3 * math.pi * x[1:])) ** 2))
            + (x[-1] - 1) ** 2 * (1 + (np.sin(2 * math.pi * x[-1])) ** 2)
        )
        + np.sum(_ufun(x, 5, 100, 4))
    )


# ── Multimodal Benchmark Functions ────────────────────────────────────────────

def F6(x):
    """Rastrigin function. Global min = 0."""
    dim = len(x)
    return 10 * dim + np.sum(x ** 2 - 10 * np.cos(2 * math.pi * x))


def F7(x):
    """Rosenbrock (banana) function. Global min = 0."""
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


def F8(x):
    """Schwefel 2.26. Global min ≈ 0 (achieved at x_i ≈ 420.9687)."""
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def F9(x):
    """Step function. Global min = 0."""
    return np.sum((np.floor(x + 0.5)) ** 2)


# ── Benchmark metadata: (function, lower bound, upper bound, global minimum) ──

BENCHMARKS = {
    "F1": (F1, -100,  100,  0),
    "F2": (F2,  -10,   10,  0),
    "F3": (F3,  -32,   32,  0),
    "F4": (F4, -600,  600,  0),
    "F5": (F5,  -50,   50,  0),
    "F6": (F6,  -5.12, 5.12, 0),
    "F7": (F7,  -30,   30,  0),
    "F8": (F8, -500,  500,  0),
    "F9": (F9, -100,  100,  0),
}
