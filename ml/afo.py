"""Archerfish Optimization (AFO) for wrapper-based feature selection.

Biological motivation
---------------------
Archerfish hunt land-based insects by shooting a jet of water through the
air-water interface.  Because light bends at that interface, the fish must aim
not at the *apparent* position of the prey but at a refraction-corrected point.
A school of archerfish also competes for a single prey, and the jet's spread
determines whether a shot hits.

Algorithmic model (implementation assumptions)
---------------------------------------------
The exact equations of the AFO variant referenced in the project report are not
published in the report itself, so this module implements a transparent
Archerfish-inspired population metaheuristic.  Each archer is a continuous
position ``x in [0, 1]^d`` where ``d`` is the number of clinical features.  A
position is decoded into a feature subset by thresholding at 0.5 (with a
minimum of 2 and a maximum of d-1 selected features, so that the subset stays
non-degenerate).

For iteration ``t = 1..T`` and archer ``i``::

    rho_i(t)  = 1 / (1 + beta * |f_i(t) - f_best(t)|)      # aim correction
    target    = peer (with probability p_c) or prey (best position)
    v_i(t+1)  = w * v_i(t)
                + c1 * r1 * (target - x_i(t)) * rho_i(t)
                + c2 * r2 * (x_best - x_i(t)) * (1 - rho_i(t))
                + eta(t) * N(0, 1)
    x_i(t+1)  = surface_reflect(x_i(t) + v_i(t+1))          # stay in [0,1]
    eta(t)    = eta0 * (1 - t / T)                          # jet dispersion

``p_c`` is a small competition probability that lets an archer be distracted
by a school-mate instead of the prey (exploration); ``eta`` is the water-jet
dispersion which cools down over time (exploitation).  Positions outside the
unit hypercube are reflected at the "water surface" boundary rather than simply
clipped.  Greedy retention keeps a new position only if it improves fitness.

Fitness function
----------------
For a candidate subset ``S`` we run ``k``-fold stratified cross-validation with
a fast logistic-regression surrogate on the training split and minimise::

    Q(S) = 0.5 * mean_cv_f1(S) + 0.5 * mean_cv_auc(S)      # quality
    J(S) = (1 - alpha) * (1 - Q(S)) + alpha * (|S| / d)

F1 captures thresholded screening balance (precision/recall), while AUC
captures ranking quality independently of the 0.5 cut-off; averaging them makes
the wrapper less sensitive to a single unstable threshold.  The second term
penalises dimensionality, so ``alpha`` trades model quality against parsimony.  Every candidate mask is cached,
and the logistic surrogate keeps the wrapper search fast enough for a live
prototype; the final SVM / Random Forest / KNN / Naive Bayes models are then
trained on the winning subset and evaluated on the untouched 20% test set.

All randomness is seeded, so a run is reproducible.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ArcherfishFeatureSelector:
    """Archerfish-inspired wrapper feature selector (minimisation)."""

    def __init__(
        self,
        n_archers: int = 22,
        n_iterations: int = 12,
        alpha: float = 0.02,
        cv_folds: int = 3,
        inertia: float = 0.55,
        attraction: float = 1.25,
        peer_pull: float = 0.55,
        dispersion: float = 0.35,
        refraction: float = 6.0,
        competition: float = 0.22,
        objective: str = "balanced",
        surrogate: str = "forest",
        min_features: int = 2,
        max_features: Optional[int] = None,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_archers = int(n_archers)
        self.n_iterations = int(n_iterations)
        self.alpha = float(alpha)
        self.cv_folds = int(cv_folds)
        self.inertia = float(inertia)
        self.attraction = float(attraction)
        self.peer_pull = float(peer_pull)
        self.dispersion = float(dispersion)
        self.refraction = float(refraction)
        self.competition = float(competition)
        if objective not in ("balanced", "f1", "auc"):
            raise ValueError("objective must be 'balanced', 'f1' or 'auc'")
        self.objective = objective
        if surrogate not in ("logistic", "forest"):
            raise ValueError("surrogate must be 'logistic' or 'forest'")
        self.surrogate = surrogate
        self.min_features = int(min_features)
        self.max_features = max_features
        self.random_state = int(random_state)
        self.verbose = verbose

        self.rng = np.random.default_rng(self.random_state)
        self._cache: Dict[Tuple[int, ...], Tuple[float, float, float, float]] = {}
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self.history: List[Dict] = []

    # ------------------------------------------------------------------ #
    # Decoding & fitness
    # ------------------------------------------------------------------ #
    def _decode(self, position: np.ndarray) -> Tuple[int, ...]:
        """Map a continuous archer position to a valid feature mask."""
        dimension = position.shape[0]
        max_features = self.max_features or max(1, dimension - 1)
        max_features = max(self.min_features, min(max_features, dimension))
        mask = position >= 0.5

        selected = int(mask.sum())
        if selected < self.min_features:
            order = np.argsort(position)[::-1][: self.min_features]
            mask = np.zeros(dimension, dtype=bool)
            mask[order] = True
        elif selected > max_features:
            chosen = np.argsort(np.where(mask, position, -np.inf))[::-1][:max_features]
            mask = np.zeros(dimension, dtype=bool)
            mask[chosen] = True
        return tuple(int(v) for v in mask)

    def _fitness(self, mask: Tuple[int, ...]) -> Tuple[float, float, float, float]:
        """Return (fitness, mean_cv_f1, mean_cv_auc, mean_cv_accuracy)."""
        if mask in self._cache:
            return self._cache[mask]

        columns = np.flatnonzero(mask)
        X = self._X[:, columns]
        y = self._y
        folds = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        f1_scores: List[float] = []
        auc_scores: List[float] = []
        acc_scores: List[float] = []
        for train_idx, valid_idx in folds.split(X, y):
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", self._make_surrogate()),
                ]
            )
            pipeline.fit(X[train_idx], y[train_idx])
            predictions = pipeline.predict(X[valid_idx])
            probabilities = pipeline.predict_proba(X[valid_idx])[:, 1]
            f1_scores.append(f1_score(y[valid_idx], predictions, zero_division=0))
            auc_scores.append(roc_auc_score(y[valid_idx], probabilities))
            acc_scores.append(accuracy_score(y[valid_idx], predictions))

        mean_f1 = float(np.mean(f1_scores))
        mean_auc = float(np.mean(auc_scores))
        mean_acc = float(np.mean(acc_scores))
        if self.objective == "f1":
            quality = mean_f1
        elif self.objective == "auc":
            quality = mean_auc
        else:
            quality = 0.5 * mean_f1 + 0.5 * mean_auc
        size_penalty = len(columns) / float(len(mask))
        fitness = (1.0 - self.alpha) * (1.0 - quality) + self.alpha * size_penalty
        result = (float(fitness), mean_f1, mean_auc, mean_acc)
        self._cache[mask] = result
        return result

    def _make_surrogate(self):
        """Fast classifier used inside the AFO wrapper objective."""
        if self.surrogate == "forest":
            return RandomForestClassifier(
                n_estimators=80,
                max_depth=6,
                min_samples_leaf=3,
                random_state=self.random_state,
                n_jobs=1,
            )
        return LogisticRegression(
            C=1.0,
            max_iter=500,
            solver="liblinear",
            random_state=self.random_state,
        )

    def _seed_population(self, dimension: int) -> np.ndarray:
        """Initialise the school, with one archer positioned to keep all features."""
        school = self.rng.uniform(0.0, 1.0, size=(self.n_archers, dimension))

        # One informed archer starts at the all-feature corner.  Because the
        # decoded subset is capped at d - 1 features, this is effectively a
        # near-full starting point rather than the exact 8-feature baseline.
        school[0, :] = 1.0

        return school

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
    ) -> Dict:
        """Run AFO and return the best subset plus the optimisation trace."""
        started = time.time()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self._X, self._y = X, y
        self._cache.clear()
        self.history = []

        if X.ndim != 2 or X.shape[1] != len(feature_names):
            raise ValueError("X columns and feature_names must match")
        if X.shape[1] < 3:
            raise ValueError("AFO feature selection needs at least 3 features")

        dimension = X.shape[1]
        school = self._seed_population(dimension)
        velocities = self.rng.normal(0.0, 0.05, size=school.shape)

        fitness = np.empty(self.n_archers)
        f1_values = np.empty(self.n_archers)
        auc_values = np.empty(self.n_archers)
        accuracy_values = np.empty(self.n_archers)
        masks: List[Tuple[int, ...]] = []
        for index in range(self.n_archers):
            mask = self._decode(school[index])
            masks.append(mask)
            (
                fitness[index],
                f1_values[index],
                auc_values[index],
                accuracy_values[index],
            ) = self._fitness(mask)

        best_index = int(np.argmin(fitness))
        best_position = school[best_index].copy()
        best_mask = masks[best_index]
        best_fitness, best_f1, best_auc, best_accuracy = self._fitness(best_mask)

        baseline_mask = tuple([1] * dimension)
        baseline_fitness, baseline_f1, baseline_auc, baseline_accuracy = self._fitness(
            baseline_mask
        )

        self.history.append(
            {
                "iteration": 0,
                "best_fitness": round(float(best_fitness), 6),
                "mean_fitness": round(float(np.mean(fitness)), 6),
                "best_cv_f1": round(float(best_f1), 4),
                "best_cv_auc": round(float(best_auc), 4),
                "best_cv_accuracy": round(float(best_accuracy), 4),
                "selected_count": int(sum(best_mask)),
            }
        )

        for iteration in range(1, self.n_iterations + 1):
            cooling = 1.0 - (iteration / float(self.n_iterations))
            dispersion = self.dispersion * max(0.08, cooling)
            progress = iteration / float(self.n_iterations)
            competition = self.competition * (1.0 - progress) + 0.04

            for index in range(self.n_archers):
                if index == best_index:
                    # The leading fish occasionally re-aims broadly to escape
                    # a local optimum (water-jet spread around the prey).
                    candidate = best_position + self.rng.normal(
                        0.0, 0.10 * cooling + 0.02, size=dimension
                    )
                else:
                    # Aim correction: archers whose fitness is close to the
                    # prey take a more precise, prey-directed shot.
                    aim_correction = 1.0 / (
                        1.0 + self.refraction * abs(fitness[index] - best_fitness)
                    )
                    if self.rng.random() < competition:
                        peer = int(self.rng.integers(0, self.n_archers))
                        target = school[peer]
                    else:
                        target = best_position

                    r1, r2 = self.rng.random(2)
                    velocity = (
                        self.inertia * velocities[index]
                        + self.attraction
                        * r1
                        * (target - school[index])
                        * aim_correction
                        + self.peer_pull
                        * r2
                        * (best_position - school[index])
                        * (1.0 - aim_correction)
                        + dispersion * self.rng.normal(0.0, 1.0, size=dimension)
                    )
                    candidate = school[index] + velocity

                # Reflect off the water surface boundary, then clamp.
                candidate = np.abs(candidate)
                candidate = np.where(candidate > 1.0, 2.0 - candidate, candidate)
                candidate = np.clip(candidate, 0.0, 1.0)

                candidate_mask = self._decode(candidate)
                (candidate_fitness, candidate_f1, candidate_auc, candidate_accuracy) = (
                    self._fitness(candidate_mask)
                )

                if candidate_fitness < fitness[index] - 1e-12:
                    velocities[index] = candidate - school[index]
                    school[index] = candidate
                    masks[index] = candidate_mask
                    fitness[index] = candidate_fitness
                    f1_values[index] = candidate_f1
                    auc_values[index] = candidate_auc
                    accuracy_values[index] = candidate_accuracy

                    if candidate_fitness < best_fitness - 1e-12:
                        best_fitness = candidate_fitness
                        best_f1 = candidate_f1
                        best_auc = candidate_auc
                        best_accuracy = candidate_accuracy
                        best_mask = candidate_mask
                        best_position = candidate.copy()
                        best_index = index

            self.history.append(
                {
                    "iteration": iteration,
                    "best_fitness": round(float(best_fitness), 6),
                    "mean_fitness": round(float(np.mean(fitness)), 6),
                    "best_cv_f1": round(float(best_f1), 4),
                    "best_cv_auc": round(float(best_auc), 4),
                    "best_cv_accuracy": round(float(best_accuracy), 4),
                    "selected_count": int(sum(best_mask)),
                }
            )
            if self.verbose:
                print(
                    "  AFO iteration {:>2}/{}  best J={:.4f}  CV-F1={:.4f}  "
                    "CV-AUC={:.4f}  features={}".format(
                        iteration,
                        self.n_iterations,
                        best_fitness,
                        best_f1,
                        best_auc,
                        sum(best_mask),
                    )
                )

        # Consensus of the final school: how often each clinical attribute is
        # included.  Used for the UI's "selection frequency" chart.
        frequency = np.zeros(dimension)
        for mask in masks:
            frequency += np.asarray(mask)
        frequency /= float(self.n_archers)

        selected = [feature_names[i] for i, value in enumerate(best_mask) if value]
        excluded = [feature_names[i] for i, value in enumerate(best_mask) if not value]
        runtime = time.time() - started

        return {
            "selected_features": selected,
            "excluded_features": excluded,
            "mask": [int(v) for v in best_mask],
            "selected_count": len(selected),
            "total_features": dimension,
            "best_fitness": round(float(best_fitness), 6),
            "best_cv_f1": round(float(best_f1), 4),
            "best_cv_auc": round(float(best_auc), 4),
            "best_cv_accuracy": round(float(best_accuracy), 4),
            "baseline_fitness": round(float(baseline_fitness), 6),
            "baseline_cv_f1": round(float(baseline_f1), 4),
            "baseline_cv_auc": round(float(baseline_auc), 4),
            "baseline_cv_accuracy": round(float(baseline_accuracy), 4),
            "improvement": round(float(baseline_fitness - best_fitness), 6),
            "fitness_gain_pct": round(
                float(
                    100.0
                    * (baseline_fitness - best_fitness)
                    / baseline_fitness
                )
                if baseline_fitness
                else 0.0,
                2,
            ),
            "feature_frequency": {
                name: round(float(frequency[i]), 3)
                for i, name in enumerate(feature_names)
            },
            "convergence": self.history,
            "n_archers": self.n_archers,
            "n_iterations": self.n_iterations,
            "alpha": self.alpha,
            "objective": self.objective,
            "surrogate": self.surrogate,
            "cv_folds": self.cv_folds,
            "evaluations": len(self._cache),
            "runtime_seconds": round(runtime, 2),
            "seed": self.random_state,
        }


def selection_fitness(
    f1: float,
    auc: float,
    selected_count: int,
    total_features: int,
    alpha: float = 0.02,
) -> float:
    """Expose the documented balanced objective for tests/documentation."""
    quality = 0.5 * f1 + 0.5 * auc
    return (1.0 - alpha) * (1.0 - quality) + alpha * (selected_count / float(total_features))
