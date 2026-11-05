"""Pareto utilities (all objectives are minimised), following notebook Step 6.63."""

import numpy as np


def dominates(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(objectives) -> list:
    """Fast non-dominated sorting. Returns a list of fronts (lists of indices)."""
    objectives = np.asarray(objectives, dtype=float)
    n = len(objectives)
    dominated_by = [[] for _ in range(n)]
    counts = np.zeros(n, dtype=int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominated_by[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                counts[p] += 1
        if counts[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                counts[q] -= 1
                if counts[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


def crowding_distance(objectives) -> np.ndarray:
    objectives = np.asarray(objectives, dtype=float)
    n, m = objectives.shape if objectives.ndim == 2 else (len(objectives), 0)
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    for j in range(m):
        order = np.argsort(objectives[:, j])
        dist[order[0]] = dist[order[-1]] = np.inf
        span = objectives[order[-1], j] - objectives[order[0], j]
        if span == 0:
            continue
        for k in range(1, n - 1):
            dist[order[k]] += (objectives[order[k + 1], j] - objectives[order[k - 1], j]) / span
    return dist


def update_archive(archive: list, candidates: list, max_size: int = 30) -> list:
    """Merge candidate dicts (with an ``objectives`` key) into a bounded non-dominated archive."""
    pool = {tuple(np.round(c["objectives"], 12)) + (c.get("key"),): c for c in archive + candidates}
    pool = list(pool.values())
    objs = np.array([c["objectives"] for c in pool])
    front = non_dominated_sort(objs)[0]
    kept = [pool[i] for i in front]
    if len(kept) > max_size:
        cd = crowding_distance(np.array([c["objectives"] for c in kept]))
        kept = [kept[i] for i in np.argsort(-cd)[:max_size]]
    return kept
