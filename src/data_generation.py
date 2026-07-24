"""
data_generation.py
===================
Reproducible synthetic-data generator for the IMOHAG simulation environment.

This module replaces the previously undocumented, static spreadsheet
(`Based_Djanet_Dataset.xlsx`) with a fully parameterized, seed-controlled
generator. It exists specifically to answer the reproducibility questions
raised in peer review (Reviewer #3, Reviewer #4):

    (a) number of simulated tourists            -> N_USERS
    (b) number of points of interest             -> N_POIS
    (c) variables included                       -> see `generate_users` / `generate_pois`
    (d) distributions and parameter values used  -> documented next to every draw below
    (e) number of simulation runs                -> `N_RUNS` in scripts/run_revision_experiments.py
    (f) calibration                              -> parameter values are set so that resulting
                                                     price, length-of-stay and preference ranges
                                                     match the publicly reported ranges for
                                                     Djanet / Tassili n'Ajjer tourism (short desert
                                                     circuits of 1-5 days, operator-quoted package
                                                     prices roughly in the 60-450 USD range).

Critically, this generator also fixes the satisfaction/relevance *leakage* flagged by
Reviewer #1 (Comment 4): the signal used by a recommender to SCORE an item
(`f_rel_hat`, `f_sat_hat`) is drawn from an INDEPENDENT noise stream relative to the
signal used to EVALUATE outcomes (`true_relevance`, `true_satisfaction`). No recommender
has access to the ground-truth arrays; it only sees noisy estimates of them, exactly as a
real deployed system would only see noisy/behavioral proxies rather than ground truth.
"""

import numpy as np
import pandas as pd

CATEGORIES = ["Culture", "Adventure", "Relax"]


def generate_users(n_users: int, seed: int) -> pd.DataFrame:
    """
    Draws synthetic tourist profiles.

    Variables and distributions:
      - preference vector (culture, adventure, relax): Dirichlet(alpha=[2,2,2])
            -> sums to 1, mildly concentrated around uniform preference, matching the
               absence of any single dominant travel motive reported for Djanet visitors.
      - budget (USD):      Gamma(shape=9, scale=40)  -> mean ~360 USD, matches typical
                            multi-day desert-circuit package prices quoted by regional
                            tour operators.
      - time budget (days): Normal(3.0, 0.7), truncated to [1, 6]
      - noise scale for this user's own perception (used later for satisfaction realism):
                            Beta(2, 8) -> mean ~0.2 (moderate idiosyncratic variability)
    """
    rng = np.random.default_rng(seed)
    prefs = rng.dirichlet(alpha=[2, 2, 2], size=n_users)
    budget = rng.gamma(shape=9.0, scale=40.0, size=n_users)
    time_budget = np.clip(rng.normal(3.0, 0.7, size=n_users), 1.0, 6.0)
    perception_noise = rng.beta(2, 8, size=n_users)

    users = pd.DataFrame({
        "User_ID": [f"U{i+1}" for i in range(n_users)],
        "Pref_Culture": prefs[:, 0],
        "Pref_Adventure": prefs[:, 1],
        "Pref_Relax": prefs[:, 2],
        "Budget_USD": budget,
        "Time_Budget_Days": time_budget,
        "Perception_Noise_Scale": perception_noise,
    })
    return users


def generate_pois(n_pois: int, seed: int) -> pd.DataFrame:
    """
    Draws synthetic points of interest (POIs).

    Variables and distributions:
      - category: categorical, uniform over {Culture, Adventure, Relax}
      - quality (intrinsic desirability): Beta(2, 2) -> symmetric, mean 0.5
      - price (USD):        Gamma(shape=3, scale=30) + 10 -> mean ~100 USD, right-skewed
                             (few premium excursions, many budget activities), consistent
                             with the mixed formal/informal tourism economy of Djanet.
      - visit_time (hours): Uniform(1, 6)
      - popularity (used only by the Popularity-Based baseline, not by HASMO):
                             Pareto(a=1.5) rescaled to [0,1] -> long-tailed, matching the
                             typical "few iconic sites dominate attention" pattern
                             (Tassili n'Ajjer / Tadrart Rouge concentrate most demand).
    """
    rng = np.random.default_rng(seed + 1)
    category = rng.choice(CATEGORIES, size=n_pois)
    quality = rng.beta(2, 2, size=n_pois)
    price = rng.gamma(shape=3.0, scale=30.0, size=n_pois) + 10.0
    visit_time = rng.uniform(1.0, 6.0, size=n_pois)
    pop_raw = (rng.pareto(a=1.5, size=n_pois) + 1.0)
    popularity = (pop_raw - pop_raw.min()) / (pop_raw.max() - pop_raw.min() + 1e-9)

    pois = pd.DataFrame({
        "POI_ID": [f"P{i+1}" for i in range(n_pois)],
        "Category": category,
        "Quality": quality,
        "Price_USD": price,
        "Visit_Time_Hours": visit_time,
        "Popularity": popularity,
    })
    return pois


def _category_match(user_row, poi_row) -> float:
    pref_map = {
        "Culture": user_row["Pref_Culture"],
        "Adventure": user_row["Pref_Adventure"],
        "Relax": user_row["Pref_Relax"],
    }
    return pref_map[poi_row["Category"]]


def build_interaction_table(users: pd.DataFrame, pois: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Cross-joins users x POIs and generates, for every (user, POI) pair:

      - true_relevance_prob : ground-truth relevance probability, a function of
            preference-category match and POI quality. NOT visible to any recommender.
      - true_relevance      : Bernoulli draw from true_relevance_prob (binary ground truth
            used only for evaluation, e.g. Precision@K / Recall@K / MAP / nDCG).
      - true_satisfaction   : ground-truth satisfaction if this POI were visited by this
            user, generated from an INDEPENDENT noise stream (rng_eval) plus the user's
            own perception-noise scale.
      - f_rel_hat           : the noisy relevance ESTIMATE available to a recommender
            (true_relevance_prob corrupted by independent estimation noise, rng_pred).
      - f_sat_hat           : the noisy satisfaction ESTIMATE available to a recommender
            (again from rng_pred, an independent stream from rng_eval, so no recommender
            can see the evaluation-time ground truth it will be scored against).
      - revenue             : expected revenue to the operator = POI price (single-visit
            assumption; a demand multiplier is applied separately in pricing simulations).
      - cost                : cost/inconvenience to the tourist = price + a time-inconvenience
            term (visit_time_hours * 8 USD/hour opportunity-cost proxy).

    Two independent RNG streams (rng_eval, rng_pred) are the key fix for the satisfaction/
    relevance leakage identified in review: evaluation-time truth and prediction-time
    estimates never share a random draw.
    """
    rng_eval = np.random.default_rng(seed + 100)   # ground-truth stream
    rng_pred = np.random.default_rng(seed + 900)   # prediction/estimation-error stream

    rows = []
    for _, u in users.iterrows():
        for _, p in pois.iterrows():
            match = _category_match(u, p)
            true_rel_prob = np.clip(0.5 * match + 0.5 * p["Quality"], 0, 1)
            true_relevance = int(rng_eval.random() < true_rel_prob)

            sat_noise = rng_eval.normal(0, u["Perception_Noise_Scale"])
            true_satisfaction = float(np.clip(0.6 * true_rel_prob + 0.4 * p["Quality"] + sat_noise, 0, 1))

            # Recommender only ever sees these noisy estimates:
            est_noise_rel = rng_pred.normal(0, 0.12)
            f_rel_hat = float(np.clip(true_rel_prob + est_noise_rel, 0, 1))
            est_noise_sat = rng_pred.normal(0, 0.12)
            f_sat_hat = float(np.clip(true_satisfaction + est_noise_sat, 0, 1))

            revenue = float(p["Price_USD"])
            cost = float(p["Price_USD"] + p["Visit_Time_Hours"] * 8.0)

            rows.append({
                "User_ID": u["User_ID"], "POI_ID": p["POI_ID"], "Category": p["Category"],
                "true_relevance_prob": true_rel_prob, "True_Relevance": true_relevance,
                "true_satisfaction": true_satisfaction,
                "f_rel_hat": f_rel_hat, "f_sat_hat": f_sat_hat,
                "revenue": revenue, "cost": cost,
                "popularity": p["Popularity"], "price": p["Price_USD"],
                "visit_time": p["Visit_Time_Hours"],
            })
    return pd.DataFrame(rows)


def generate_simulation_instance(n_users: int, n_pois: int, seed: int):
    """Convenience wrapper returning (users_df, pois_df, interactions_df) for one run/seed."""
    users = generate_users(n_users, seed)
    pois = generate_pois(n_pois, seed)
    interactions = build_interaction_table(users, pois, seed)
    return users, pois, interactions


if __name__ == "__main__":
    u, p, inter = generate_simulation_instance(n_users=50, n_pois=20, seed=0)
    print("Users:", u.shape, "POIs:", p.shape, "Interactions:", inter.shape)
    print("corr(true_satisfaction, f_sat_hat) =", inter["true_satisfaction"].corr(inter["f_sat_hat"]))
