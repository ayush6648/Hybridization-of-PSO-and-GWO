# Hybridization of PSO and GWO (HPSOGWO)

A Python implementation of the **Hybrid Particle Swarm Optimization and Grey Wolf Optimizer (HPSOGWO)**, benchmarked against the standalone PSO and GWO algorithms.

Based on the research paper:
> Singh, N., & Singh, S. B. (2017). **Hybrid Algorithm of Particle Swarm Optimization and Grey Wolf Optimizer for Improving Convergence Performance**. *Journal of Applied Mathematics*, 2017, 1–15.

---

## Overview

The core idea is to combine:
- **PSO's exploitation strength** — particles converge fast toward known good solutions.
- **GWO's exploration strength** — the wolf hierarchy avoids premature convergence.

The hybrid (HPSOGWO) uses GWO's alpha, beta, and delta wolves to guide the PSO velocity update, resulting in better convergence in terms of both solution quality and speed.

---

## Project Structure

```
.
├── benchmarks.py   # 9 standard benchmark functions with bounds metadata
├── GWO.py          # Grey Wolf Optimizer implementation
├── HPSOGWO.py      # Hybrid PSO-GWO implementation
├── PSO.py          # Particle Swarm Optimization implementation
├── plot.py         # CLI entry point — runs all three and plots results
├── requirements.txt
└── .gitignore
```

---

## Benchmark Functions

| Name | Function         | Search Space        | Global Min |
|------|------------------|---------------------|-----------|
| F1   | Sphere           | [-100, 100]         | 0         |
| F2   | Schwefel 2.22    | [-10, 10]           | 0         |
| F3   | Ackley           | [-32, 32]           | 0         |
| F4   | Griewank         | [-600, 600]         | 0         |
| F5   | Penalized Levy   | [-50, 50]           | 0         |
| F6   | Rastrigin        | [-5.12, 5.12]       | 0         |
| F7   | Rosenbrock       | [-30, 30]           | 0         |
| F8   | Schwefel 2.26    | [-500, 500]         | 0         |
| F9   | Step             | [-100, 100]         | 0         |

---

## Installation

```bash
git clone https://github.com/your-username/Hybridization-of-PSO-and-GWO.git
cd Hybridization-of-PSO-and-GWO
pip install -r requirements.txt
```

> Python 3.9+ is recommended.

---

## How to Run

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/Hybridization-of-PSO-and-GWO.git
cd Hybridization-of-PSO-and-GWO
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the comparison plot

```bash
python plot.py
```

This runs all three algorithms (GWO, HPSOGWO, PSO) on the **Sphere** benchmark (F1 by default) and opens an interactive plot showing convergence curves.

### Step 4 — Try different benchmark functions

```bash
# List all available functions with their bounds
python plot.py --list

# Run on Ackley (F3)
python plot.py --func F3

# Run on Rastrigin (F6) with a fixed random seed
python plot.py --func F6 --seed 42
```

---

## Usage

### Run a comparison plot

```bash
# Compare on the Sphere function (default)
python plot.py

# Compare on a specific benchmark
python plot.py --func F3

# Fix the random seed for reproducibility
python plot.py --func F5 --seed 42

# List all available benchmark functions
python plot.py --list
```

### Use the algorithms directly

```python
import numpy as np
from HPSOGWO import HPSOGWO
from GWO import GWO
from PSO import PSO
import benchmarks

func, lb, ub, _ = benchmarks.BENCHMARKS["F1"]

model = HPSOGWO(func, num_dim=30, pop_size=20, max_iter=100, lb=lb, ub=ub)
model.opt()
result = model.return_result()
print("Best score:", result[-1])
```

---

## Results

HPSOGWO consistently outperforms both GWO and PSO across the 9 benchmark functions, either:
- **reaching the global optimum** in fewer iterations, or
- **achieving a lower best-score** when both converge to sub-optimal solutions.

The convergence curve is plotted as **Best Score vs. Number of Iterations**.

---

## Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_dim` | 30 | Problem dimensionality |
| `pop_size` | 20 | Population / swarm size |
| `max_iter` | 100 | Maximum iterations |
| `lb` / `ub` | function-specific | Search space bounds |
| `c1`, `c2`, `c3` | 0.5 | Acceleration coefficients |
| `init_w` | 0.5 | Initial inertia weight |
| `a_max` / `a_min` | 2 / 0 | Linear decrease parameter |
| `k` | 0.2 | Velocity clamping factor |

---

## License

This project is released for academic and educational purposes.
