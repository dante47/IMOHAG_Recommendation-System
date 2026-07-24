"""
hasmo.py
========
Implements the HASMO scoring function (manuscript Eq. 6), the constrained
top-K selection procedure (manuscript Algorithm 1), a weight-sensitivity sweep,
and a compact NSGA-II used purely as a Pareto-front reference point.

Terminology note (Reviewer #1 Comment 2 / Reviewer #4 Comment 3):
HASMO is a *weighted-scalarization* decision procedure, not a general-purpose
multi-objective optimizer. It is described and implemented as such here. The
NSGA-II routine below is provided only as a theoretical reference point against
which the scalarized solution can be situated on the trade-off surface -- it is
NOT the deployed recommendation algorithm (see README, "Implementation Status").
"""

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {"rel": 0.35, "sat": 0.35, "rev": 0.20, "cost": 0.10}


def _normalize01(x: np.ndarray) -> np.ndarray:
    lo, hi = np.min(x), np.max(x)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def hasmo_score(interactions: pd.DataFrame, weights: dict = None) -> pd.Series:
    """
    Score_{u,i} = lambda1 * f_rel_hat + lambda2 * f_sat_hat + lambda3 * revenue_norm
                  - lambda4 * cost_norm

    Revenue and cost are min-max normalized per call so that weights are comparable
    regardless of the raw USD scale (documented explicitly, addressing Reviewer #3's
    request to clarify how the scalarization combines heterogeneous units).
    """
    w = weights or DEFAULT_WEIGHTS
    rev_norm = _normalize01(interactions["revenue"].values)
    cost_norm = _normalize01(interactions["cost"].values)
    score = (
        w["rel"] * interactions["f_rel_hat"].values
        + w["sat"] * interactions["f_sat_hat"].values
        + w["rev"] * rev_norm
        - w["cost"] * cost_norm
    )
    return pd.Series(score, index=interactions.index)


def greedy_topk_selection(interactions: pd.DataFrame, users: pd.DataFrame, score_col: str,
                           top_k: int = 5) -> pd.DataFrame:
    """
    Algorithm 1: for each user, sort candidate POIs by `score_col` descending and greedily
    add items while respecting the user's budget and time constraints, stopping at `top_k`
    selections. Returns a long-format DataFrame of selected (User_ID, POI_ID) rows.

    Complexity: O(m * n) to compute scores for all user-POI pairs (already materialized in
    `interactions`), plus O(n log n) sorting per user -> O(m * n log n) overall, as noted
    in response to Reviewer #1's request for a complexity discussion.
    """
    budgets = users.set_index("User_ID")["Budget_USD"]
    time_budgets = users.set_index("User_ID")["Time_Budget_Days"] * 8.0  # hours available

    selected_rows = []
    for user_id, group in interactions.groupby("User_ID"):
        g = group.sort_values(score_col, ascending=False)
        b_remaining = budgets[user_id]
        t_remaining = time_budgets[user_id]
        n_selected = 0
        for _, row in g.iterrows():
            if n_selected >= top_k:
                break
            if row["price"] <= b_remaining and row["visit_time"] <= t_remaining:
                selected_rows.append(row)
                b_remaining -= row["price"]
                t_remaining -= row["visit_time"]
                n_selected += 1
    return pd.DataFrame(selected_rows)


def weight_sensitivity_sweep(users, interactions, sat_weight_grid=np.linspace(0.0, 1.0, 11), top_k=5):
    """
    Varies lambda_sat over `sat_weight_grid`; the remaining weight budget (1 - lambda_sat)
    is split equally across relevance, revenue, and cost, and the resulting selections are
    evaluated. Returns a tidy DataFrame with one row per grid point, reporting the mean
    achieved relevance, satisfaction, revenue, and cost of the selected items.

    This directly answers the sensitivity-analysis request from Reviewer #1 (Comment 2),
    Reviewer #3 (Comment 2), and Reviewer #4 (Comment 3).
    """
    rows = []
    for lam_sat in sat_weight_grid:
        remaining = 1.0 - lam_sat
        w = {"rel": remaining / 3, "sat": lam_sat, "rev": remaining / 3, "cost": remaining / 3}
        scores = hasmo_score(interactions, w)
        sel = greedy_topk_selection(interactions.assign(_score=scores), users, "_score", top_k=top_k)
        if len(sel) == 0:
            continue
        rows.append({
            "lambda_sat": lam_sat,
            "mean_true_relevance_prob": sel["true_relevance_prob"].mean(),
            "mean_true_satisfaction": sel["true_satisfaction"].mean(),
            "mean_revenue": sel["revenue"].mean(),
            "mean_cost": sel["cost"].mean(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Compact NSGA-II over weight vectors (lambda1..lambda4), used ONLY to produce
# a Pareto-front reference point for comparison against the scalarized HASMO
# solution. Objectives: maximize mean relevance, mean satisfaction, mean revenue;
# minimize mean cost, evaluated via the same greedy top-K selection procedure.
# ---------------------------------------------------------------------------

def _evaluate_weight_vector(w_vec, users, interactions, top_k=5):
    w = {"rel": w_vec[0], "sat": w_vec[1], "rev": w_vec[2], "cost": w_vec[3]}
    scores = hasmo_score(interactions, w)
    sel = greedy_topk_selection(interactions.assign(_score=scores), users, "_score", top_k=top_k)
    if len(sel) == 0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.array([
        sel["true_relevance_prob"].mean(),
        sel["true_satisfaction"].mean(),
        sel["revenue"].mean() / (interactions["revenue"].max() + 1e-9),
        sel["cost"].mean() / (interactions["cost"].max() + 1e-9),
    ])


def _dominates(obj_a, obj_b):
    # obj = [rel, sat, rev, cost]; first 3 maximize, cost minimizes.
    a = np.array([obj_a[0], obj_a[1], obj_a[2], -obj_a[3]])
    b = np.array([obj_b[0], obj_b[1], obj_b[2], -obj_b[3]])
    return np.all(a >= b) and np.any(a > b)


def _fast_non_dominated_front(objectives):
    n = len(objectives)
    front = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i != j and _dominates(objectives[j], objectives[i]):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def run_mini_nsga2(users, interactions, pop_size=30, generations=15, seed=0, top_k=5):
    """
    A deliberately compact NSGA-II-style search over the simplex of weight vectors
    (lambda1..lambda4, renormalized to sum to 1 after mutation). Not a production-grade
    implementation -- intended purely as a theoretical Pareto-front reference point
    contextualizing the scalarized HASMO solution, per Reviewer #1's request for
    Pareto-based analysis as an alternative/complement to weighted scalarization.
    """
    rng = np.random.default_rng(seed)
    population = rng.dirichlet(alpha=[1, 1, 1, 1], size=pop_size)

    for _ in range(generations):
        objectives = [_evaluate_weight_vector(w, users, interactions, top_k) for w in population]
        front_idx = _fast_non_dominated_front(objectives)
        parents = population[front_idx] if len(front_idx) > 0 else population
        # simple mutation/crossover: perturb parents (with replacement) to refill population
        children = []
        for _ in range(pop_size):
            p = parents[rng.integers(0, len(parents))]
            mutation = rng.normal(0, 0.08, size=4)
            child = np.clip(p + mutation, 1e-3, None)
            child = child / child.sum()
            children.append(child)
        population = np.array(children)

    objectives = [_evaluate_weight_vector(w, users, interactions, top_k) for w in population]
    front_idx = _fast_non_dominated_front(objectives)
    result = pd.DataFrame(population[front_idx], columns=["lambda_rel", "lambda_sat", "lambda_rev", "lambda_cost"])
    obj_arr = np.array(objectives)[front_idx]
    result["mean_relevance"] = obj_arr[:, 0]
    result["mean_satisfaction"] = obj_arr[:, 1]
    result["mean_revenue_norm"] = obj_arr[:, 2]
    result["mean_cost_norm"] = obj_arr[:, 3]
    return result


if __name__ == "__main__":
    from data_generation import generate_simulation_instance
    users, pois, inter = generate_simulation_instance(50, 20, seed=0)
    s = hasmo_score(inter)
    sel = greedy_topk_selection(inter.assign(_score=s), users, "_score", top_k=5)
    print("Selected rows:", len(sel), "mean satisfaction:", sel["true_satisfaction"].mean())
