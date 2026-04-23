"""
data/instance_generator.py

Generates synthetic MIP benchmark instances for the four problem classes
used in Branch Ranking and Gasse et al.
"""

import os
import numpy as np
from pyscipopt import Model


def generate_instances(problem_type, difficulty, n_instances, out_dir, seed=0):
    """
    Generate MIP instances and save as .lp files.

    Returns list of file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng   = np.random.RandomState(seed)
    paths = []

    generators = {
        "setcover":  _gen_setcover,
        "auction":   _gen_auction,
        "facility":  _gen_facility,
        "indset":    _gen_indset,
    }
    gen_fn = generators.get(problem_type)
    if gen_fn is None:
        raise ValueError(f"Unknown problem type: {problem_type}")

    # Size parameters per difficulty
    sizes = {
        "setcover":  {"easy": (500, 1000),  "medium": (500, 2000),  "hard": (500, 3000)},
        "auction":   {"easy": (100, 500),   "medium": (200, 1000),  "hard": (300, 1500)},
        "facility":  {"easy": (25, 50),     "medium": (35, 75),     "hard": (50, 100)},
        "indset":    {"easy": (500, 4),     "medium": (1000, 4),    "hard": (1500, 4)},
    }
    params = sizes[problem_type].get(difficulty, sizes[problem_type]["easy"])

    generated = 0
    existing  = sorted([f for f in os.listdir(out_dir) if f.endswith(".lp")])
    start_idx = len(existing)

    for i in range(n_instances):
        idx  = start_idx + i
        path = os.path.join(out_dir, f"{problem_type}_{difficulty}_{idx:05d}.lp")
        if os.path.exists(path):
            paths.append(path)
            generated += 1
            if generated % 100 == 0:
                print(f"  Generated {generated}/{n_instances} {problem_type} {difficulty} instances")
            continue

        instance_seed = seed * 10000 + i
        m = gen_fn(params, rng=np.random.RandomState(instance_seed))
        m.writeProblem(path)
        paths.append(path)
        generated += 1
        if generated % 100 == 0:
            print(f"  Generated {generated}/{n_instances} {problem_type} {difficulty} instances")

    print(f"  Generated {generated}/{n_instances} {problem_type} {difficulty} instances")
    return paths


# ── Problem generators ────────────────────────────────────────────────────────

def _gen_setcover(params, rng):
    """
    Set covering: minimize cost s.t. each element covered at least once.
    params = (n_rows, n_cols)
    """
    n_rows, n_cols = params
    m = Model()
    m.hideOutput(True)

    costs = rng.uniform(1, 10, n_cols)
    vars_ = [m.addVar(f"x{j}", vtype="B", obj=costs[j]) for j in range(n_cols)]
    m.setMinimize()

    density = 0.05
    for i in range(n_rows):
        covered = rng.choice(n_cols, max(1, int(n_cols * density)), replace=False)
        m.addCons(sum(vars_[j] for j in covered) >= 1, f"c{i}")

    return m


def _gen_auction(params, rng):
    """
    Combinatorial auction: maximize revenue, items assigned at most once.
    params = (n_items, n_bids)
    """
    n_items, n_bids = params
    m = Model()
    m.hideOutput(True)
    m.setMaximize()

    bids   = []
    prices = []
    for k in range(n_bids):
        bundle_size = rng.randint(1, max(2, n_items // 4))
        bundle      = rng.choice(n_items, bundle_size, replace=False)
        price       = float(rng.uniform(1, 10) * bundle_size)
        bids.append(bundle)
        prices.append(price)

    x = [m.addVar(f"x{k}", vtype="B", obj=prices[k]) for k in range(n_bids)]

    for i in range(n_items):
        covering = [x[k] for k, b in enumerate(bids) if i in b]
        if covering:
            m.addCons(sum(covering) <= 1, f"item{i}")

    return m


def _gen_facility(params, rng):
    """
    Capacitated facility location: minimize cost of opening facilities + transport.
    params = (n_facilities, n_customers)
    """
    n_fac, n_cust = params
    m = Model()
    m.hideOutput(True)
    m.setMinimize()

    fixed_cost   = rng.uniform(100, 200, n_fac)
    transp_cost  = rng.uniform(1, 10, (n_fac, n_cust))
    demand       = rng.uniform(1, 5, n_cust)
    capacity     = rng.uniform(20, 40, n_fac)

    y = [m.addVar(f"y{i}", vtype="B", obj=fixed_cost[i]) for i in range(n_fac)]
    x = [[m.addVar(f"x{i}_{j}", vtype="C", lb=0.0, ub=1.0, obj=transp_cost[i][j])
          for j in range(n_cust)] for i in range(n_fac)]

    for j in range(n_cust):
        m.addCons(sum(x[i][j] for i in range(n_fac)) >= 1, f"dem{j}")

    for i in range(n_fac):
        m.addCons(sum(demand[j] * x[i][j] for j in range(n_cust)) <= capacity[i] * y[i], f"cap{i}")

    for i in range(n_fac):
        for j in range(n_cust):
            m.addCons(x[i][j] <= y[i], f"link{i}_{j}")

    return m


def _gen_indset(params, rng):
    """
    Maximum independent set on random graph.
    params = (n_nodes, avg_degree)
    """
    n_nodes, avg_degree = params
    m = Model()
    m.hideOutput(True)
    m.setMaximize()

    x = [m.addVar(f"x{i}", vtype="B", obj=1.0) for i in range(n_nodes)]

    p = float(avg_degree) / float(n_nodes - 1)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < p:
                m.addCons(x[i] + x[j] <= 1, f"e{i}_{j}")

    return m
