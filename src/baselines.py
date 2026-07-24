"""
baselines.py
============
Baseline scoring methods compared against HASMO.

Original internal baselines (already present in the manuscript):
    - Random
    - Relevance-Only
    - Revenue-Driven
    - Static-Weighted-Sum (equal weights, no tuning/sensitivity analysis)

New baselines added in direct response to peer review (Reviewer #1 Comment 2;
Reviewer #3 Comment 5; Reviewer #4 Comment 4), which asked for comparison against
established recommendation and multi-criteria decision-making approaches rather
than internally designed baselines only:
    - Popularity-Based ranking       (classical, non-personalized recommender baseline)
    - Content-Based filtering        (cosine similarity between user preference vector
                                       and POI category vector; a personalized baseline
                                       that does not require a dense interaction matrix,
                                       and is therefore fair to compare in this sparse-data
                                       setting -- unlike collaborative filtering / matrix
                                       factorization, whose cold-start failure under near-zero
                                       interaction density is already well documented and is
                                       not re-implemented here; see README / rebuttal letter)
    - TOPSIS                         (classical multi-criteria decision-making baseline,
                                       applied over the same four criteria used by HASMO)
"""

import numpy as np
import pandas as pd
from .hasmo import hasmo_score, _normalize01


def random_score(interactions: pd.DataFrame, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.random(len(interactions)), index=interactions.index)


def relevance_only_score(interactions: pd.DataFrame) -> pd.Series:
    return interactions["f_rel_hat"].copy()


def revenue_driven_score(interactions: pd.DataFrame) -> pd.Series:
    return pd.Series(_normalize01(interactions["revenue"].values), index=interactions.index)


def static_weighted_sum_score(interactions: pd.DataFrame) -> pd.Series:
    equal_w = {"rel": 0.25, "sat": 0.25, "rev": 0.25, "cost": 0.25}
    return hasmo_score(interactions, equal_w)


def popularity_based_score(interactions: pd.DataFrame) -> pd.Series:
    """Classical non-personalized baseline: rank purely by item popularity."""
    return interactions["popularity"].copy()


def content_based_score(interactions: pd.DataFrame, users: pd.DataFrame) -> pd.Series:
    """
    Cosine similarity between the user's (culture, adventure, relax) preference vector
    and a one-hot category vector for each POI. A personalized baseline that, unlike
    collaborative filtering, does not require historical interaction density.
    """
    pref_lookup = users.set_index("User_ID")[["Pref_Culture", "Pref_Adventure", "Pref_Relax"]]
    cat_to_onehot = {
        "Culture": np.array([1, 0, 0]),
        "Adventure": np.array([0, 1, 0]),
        "Relax": np.array([0, 0, 1]),
    }
    scores = np.zeros(len(interactions))
    for pos, (_, row) in enumerate(interactions.iterrows()):
        u_vec = pref_lookup.loc[row["User_ID"]].values.astype(float)
        p_vec = cat_to_onehot[row["Category"]].astype(float)
        denom = (np.linalg.norm(u_vec) * np.linalg.norm(p_vec)) + 1e-9
        scores[pos] = float(np.dot(u_vec, p_vec) / denom)
    return pd.Series(scores, index=interactions.index)


def topsis_score(interactions: pd.DataFrame) -> pd.Series:
    """
    Classical TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
    applied per user over the criteria [f_rel_hat, f_sat_hat, revenue, cost] with equal
    criterion weights. Benefit criteria: relevance, satisfaction, revenue. Cost criterion:
    cost (to be minimized).
    """
    scores = np.zeros(len(interactions))
    for user_id, group in interactions.groupby("User_ID"):
        idx = group.index
        M = group[["f_rel_hat", "f_sat_hat", "revenue", "cost"]].values.astype(float)
        norm = np.sqrt((M ** 2).sum(axis=0))
        norm[norm == 0] = 1e-9
        M_norm = M / norm
        w = np.array([0.25, 0.25, 0.25, 0.25])
        M_w = M_norm * w
        ideal_best = np.array([M_w[:, 0].max(), M_w[:, 1].max(), M_w[:, 2].max(), M_w[:, 3].min()])
        ideal_worst = np.array([M_w[:, 0].min(), M_w[:, 1].min(), M_w[:, 2].min(), M_w[:, 3].max()])
        d_best = np.sqrt(((M_w - ideal_best) ** 2).sum(axis=1))
        d_worst = np.sqrt(((M_w - ideal_worst) ** 2).sum(axis=1))
        closeness = d_worst / (d_best + d_worst + 1e-9)
        scores[interactions.index.get_indexer(idx)] = closeness
    return pd.Series(scores, index=interactions.index)


def all_methods(interactions: pd.DataFrame, users: pd.DataFrame, seed: int = 0) -> dict:
    """
    Returns {method_name: score_series} for every method compared in the revised study.

    NOTE: `topsis_score` remains available in this module as a utility, but is
    intentionally NOT included in the reported comparison below. The manuscript's
    final response to reviewers (Response to Reviewer #4, Comment 4) commits to
    discussing AHP/TOPSIS/PROMETHEE in the Related Work section only, since IMOHAG
    is a system-level decision-support architecture rather than a pure ranking
    method and classical MCDM techniques are judged not to be directly comparable
    system-level baselines. Keep this function's method set in sync with
    Table 5 / Table 6 of the manuscript if that position changes.
    """
    return {
        "Random": random_score(interactions, seed=seed),
        "Relevance-Only": relevance_only_score(interactions),
        "Revenue-Driven": revenue_driven_score(interactions),
        "Static-Weighted-Sum": static_weighted_sum_score(interactions),
        "Popularity-Based": popularity_based_score(interactions),
        "Content-Based": content_based_score(interactions, users),
        "IMOHAG (HASMO)": hasmo_score(interactions),
    }
