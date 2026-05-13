"""
Compare GWO, HPSOGWO, and PSO on a chosen benchmark function.

Usage:
    python plot.py --func F1          # run on Sphere (default)
    python plot.py --func F5          # run on Penalized Levy
    python plot.py --list             # list all available functions
    python plot.py --func F3 --seed 0 # fixed seed for reproducibility
"""

import argparse
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import benchmarks
from GWO import GWO
from HPSOGWO import HPSOGWO
from PSO import PSO

NUM_DIM = 30
POP_SIZE = 20
MAX_ITER = 100


def run(func_name: str, seed: Optional[int] = None) -> None:
    if func_name not in benchmarks.BENCHMARKS:
        print(f"Unknown function '{func_name}'. Use --list to see available options.")
        sys.exit(1)

    func, lb, ub, optimum = benchmarks.BENCHMARKS[func_name]

    if seed is not None:
        np.random.seed(seed)

    gwo = GWO(func, num_dim=NUM_DIM, num_particle=POP_SIZE, max_iter=MAX_ITER,
              lb=lb, ub=ub)
    gwo.opt()
    gwo_result = gwo.return_result()

    hpsogwo = HPSOGWO(func, num_dim=NUM_DIM, pop_size=POP_SIZE, max_iter=MAX_ITER,
                      lb=lb, ub=ub)
    hpsogwo.opt()
    hpsogwo_result = hpsogwo.return_result()

    pso = PSO(func, dimension=NUM_DIM, pop_size=POP_SIZE, max_iter=MAX_ITER,
              lb=lb, ub=ub)
    pso.random_init()
    pso.start()
    pso_result = pso.return_result()

    plt.figure(figsize=(10, 6))
    plt.plot(gwo_result, label="GWO")
    plt.plot(hpsogwo_result, label="HPSOGWO")
    plt.plot(pso_result, label="PSO")
    plt.axhline(y=optimum, color="k", linestyle="--", linewidth=0.8, label=f"Optimum ({optimum})")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.title(f"Comparison of HPSOGWO vs PSO and GWO — {func_name} (dim={NUM_DIM})")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Best Score")
    plt.tight_layout()
    plt.show()


def list_functions() -> None:
    print(f"{'Name':<6}  {'Lower Bound':>12}  {'Upper Bound':>12}  {'Global Min':>10}  Description")
    print("-" * 70)
    descriptions = {
        "F1": "Sphere",
        "F2": "Schwefel 2.22",
        "F3": "Ackley",
        "F4": "Griewank",
        "F5": "Penalized Levy",
        "F6": "Rastrigin",
        "F7": "Rosenbrock",
        "F8": "Schwefel 2.26",
        "F9": "Step",
    }
    for name, (_, lb, ub, opt) in benchmarks.BENCHMARKS.items():
        print(f"{name:<6}  {lb:>12.2f}  {ub:>12.2f}  {opt:>10.4f}  {descriptions.get(name, '')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GWO, HPSOGWO, and PSO on standard benchmark functions."
    )
    parser.add_argument(
        "--func", default="F1", metavar="NAME",
        help="Benchmark function to use (default: F1). See --list for options."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available benchmark functions and exit."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: no fixed seed)."
    )

    args = parser.parse_args()

    if args.list:
        list_functions()
        return

    run(args.func, seed=args.seed)


if __name__ == "__main__":
    main()
