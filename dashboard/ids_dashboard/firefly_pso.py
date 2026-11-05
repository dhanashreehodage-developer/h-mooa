"""Simplified Binary Firefly-PSO feature selection used for live demonstration.

This is a reduced, fast version of the notebook's Step 6.65 algorithm (small
population, few iterations, three objectives). It is not the full original
experiment and its outputs are not research results.
"""

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

from .metrics import macro_fpr
from .pareto import non_dominated_sort, update_archive


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -50, 50)))


class FireflyPSODemo:
    def __init__(self, X_fit, y_fit, X_val, y_val, population=6, iterations=3, min_features=3,
                 trees=30, w=0.65, c1=1.5, c2=1.5, beta0=1.0, gamma=2.0, seed=42):
        self.X_fit, self.y_fit, self.X_val, self.y_val = X_fit, y_fit, X_val, y_val
        self.features = list(X_fit.columns)
        self.n = len(self.features)
        if self.n < 2:
            raise ValueError("Feature selection needs at least two features.")
        self.population = int(population)
        self.iterations = int(iterations)
        self.min_features = int(min(max(1, min_features), self.n))
        self.trees = int(trees)
        self.w, self.c1, self.c2 = w, c1, c2
        self.beta0, self.gamma = beta0, gamma
        self.rng = np.random.default_rng(seed)
        self.cache = {}
        self.labels = sorted(set(y_fit) | set(y_val))

    def _repair(self, mask):
        mask = mask.astype(int).copy()
        if mask.sum() < self.min_features:
            off = np.flatnonzero(mask == 0)
            mask[self.rng.choice(off, self.min_features - mask.sum(), replace=False)] = 1
        return mask

    def evaluate(self, mask):
        mask = self._repair(mask)
        key = tuple(mask.tolist())
        if key in self.cache:
            return self.cache[key]
        cols = [f for f, m in zip(self.features, mask) if m]
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=self.trees, class_weight="balanced", random_state=42, n_jobs=-1),
        )
        start = time.perf_counter()
        model.fit(self.X_fit[cols], self.y_fit)
        train_time = time.perf_counter() - start
        pred = model.predict(self.X_val[cols])
        f1 = f1_score(self.y_val, pred, labels=self.labels, average="macro", zero_division=0)
        fpr = macro_fpr(self.y_val, pred, self.labels)
        result = {
            "key": key,
            "mask": mask,
            "features": cols,
            "feature_count": len(cols),
            "macro_f1": float(f1),
            "macro_fpr": float(fpr),
            "training_time": train_time,
            "objectives": np.array([1 - f1, fpr, len(cols) / self.n]),
        }
        self.cache[key] = result
        return result

    def run(self, progress=None):
        pos = (self.rng.random((self.population, self.n)) < self.rng.uniform(0.2, 0.8, (self.population, 1))).astype(int)
        pos = np.array([self._repair(p) for p in pos])
        vel = self.rng.uniform(-1, 1, (self.population, self.n))
        evals = [self.evaluate(p) for p in pos]
        pbest = [e for e in evals]
        archive = update_archive([], evals)
        history = [self._snapshot(0, evals, archive)]
        total_steps = self.iterations
        for it in range(1, self.iterations + 1):
            objs = np.array([e["objectives"] for e in evals])
            ranks = np.zeros(len(evals))
            for r, front in enumerate(non_dominated_sort(objs)):
                ranks[front] = r
            for i in range(self.population):
                leader = archive[self.rng.integers(len(archive))]["mask"]
                r1, r2 = self.rng.random(self.n), self.rng.random(self.n)
                vel[i] = self.w * vel[i] + self.c1 * r1 * (pbest[i]["mask"] - pos[i]) + self.c2 * r2 * (leader - pos[i])
                brighter = [j for j in range(self.population) if ranks[j] < ranks[i]]
                if brighter:
                    j = brighter[self.rng.integers(len(brighter))]
                    distance = np.sum(pos[i] != pos[j]) / self.n
                    beta = self.beta0 * np.exp(-self.gamma * distance ** 2)
                    vel[i] += beta * (pos[j] - pos[i])
                vel[i] = np.clip(vel[i], -6, 6)
                pos[i] = self._repair((self.rng.random(self.n) < _sigmoid(vel[i])).astype(int))
            evals = [self.evaluate(p) for p in pos]
            for i, e in enumerate(evals):
                if np.all(e["objectives"] <= pbest[i]["objectives"]):
                    pbest[i] = e
            archive = update_archive(archive, evals)
            history.append(self._snapshot(it, evals, archive))
            if progress:
                progress(it / total_steps)
        return {"archive": archive, "history": history, "evaluations": list(self.cache.values())}

    def _snapshot(self, it, evals, archive):
        return {
            "iteration": it,
            "population_counts": [e["feature_count"] for e in evals],
            "population_f1": [e["macro_f1"] for e in evals],
            "archive_size": len(archive),
            "unique_evaluations": len(self.cache),
        }
