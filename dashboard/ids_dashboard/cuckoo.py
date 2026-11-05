"""Simplified Cuckoo Search over Random Forest hyperparameters (live demonstration).

A reduced version of notebook Step 6.66: Levy-flight moves in a normalised
hyperparameter space, random nest replacement, abandonment of a fraction of
nests and a Pareto archive over (1 - Macro F1, Macro FPR). It is not the
complete original experiment.
"""

import math

import numpy as np
from sklearn.metrics import f1_score

from .metrics import macro_fpr
from .modeling import build_rf
from .pareto import update_archive

MAX_FEATURES_OPTIONS = ["sqrt", "log2", 0.5, 0.8]


def levy_steps(size, beta=1.5, rng=None):
    """Mantegna's algorithm for Levy-distributed step lengths."""
    rng = rng or np.random.default_rng()
    sigma = (
        math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.normal(0, sigma, size)
    v = rng.normal(0, 1, size)
    return u / np.abs(v) ** (1 / beta)


class CuckooSearchDemo:
    def __init__(self, X_fit, y_fit, X_val, y_val, bounds, nests=5, iterations=2, pa=0.25, beta=1.5,
                 step_scale=0.1, seed=42):
        self.X_fit, self.y_fit, self.X_val, self.y_val = X_fit, y_fit, X_val, y_val
        self.bounds = bounds
        self.nests, self.iterations = int(nests), int(iterations)
        self.pa, self.beta, self.step_scale = pa, beta, step_scale
        self.rng = np.random.default_rng(seed)
        self.labels = sorted(set(y_fit) | set(y_val))
        self.cache = {}

    def decode(self, z):
        z = np.clip(z, 0, 1)
        b = self.bounds

        def scale(v, lo, hi):
            return int(round(lo + v * (hi - lo)))

        return {
            "n_estimators": scale(z[0], *b["n_estimators"]),
            "max_depth": scale(z[1], *b["max_depth"]),
            "min_samples_leaf": scale(z[2], *b["min_samples_leaf"]),
            "max_features": MAX_FEATURES_OPTIONS[min(int(z[3] * len(MAX_FEATURES_OPTIONS)), len(MAX_FEATURES_OPTIONS) - 1)],
        }

    def evaluate(self, z, stage):
        params = self.decode(z)
        key = tuple(params.values())
        if key in self.cache:
            return self.cache[key]
        model = build_rf(**params, seed=42)
        model.fit(self.X_fit, self.y_fit)
        pred = model.predict(self.X_val)
        f1 = f1_score(self.y_val, pred, labels=self.labels, average="macro", zero_division=0)
        fpr = macro_fpr(self.y_val, pred, self.labels)
        result = {
            "key": key, "z": np.clip(z, 0, 1), "params": params, "macro_f1": float(f1),
            "macro_fpr": float(fpr), "objectives": np.array([1 - f1, fpr]), "stage": stage,
            "order": len(self.cache),
        }
        self.cache[key] = result
        return result

    def run(self, progress=None):
        nests = [self.evaluate(self.rng.random(4), "Initial nest") for _ in range(self.nests)]
        archive = update_archive([], nests, max_size=20)
        log = []
        for it in range(1, self.iterations + 1):
            best = archive[self.rng.integers(len(archive))]["z"]
            for i in range(self.nests):
                step = levy_steps(4, self.beta, self.rng) * self.step_scale * (nests[i]["z"] - best + 1e-3)
                new = self.evaluate(nests[i]["z"] + step, f"Levy flight (iter {it})")
                j = self.rng.integers(self.nests)
                if np.all(new["objectives"] <= nests[j]["objectives"]):
                    nests[j] = new
                    log.append(f"Iteration {it}: Levy-flight egg replaced nest {j}.")
            n_abandon = max(1, int(round(self.pa * self.nests)))
            worst = np.argsort([n["objectives"][0] for n in nests])[-n_abandon:]
            for j in worst:
                nests[j] = self.evaluate(self.rng.random(4), f"Abandoned nest rebuilt (iter {it})")
            log.append(f"Iteration {it}: abandoned {n_abandon} worst nest(s) and built new random nests.")
            archive = update_archive(archive, nests, max_size=20)
            if progress:
                progress(it / self.iterations)
        return {"archive": archive, "evaluations": sorted(self.cache.values(), key=lambda r: r["order"]), "log": log}
