"""
stats_validation.py
====================
Statistical validation utilities added in direct response to peer review
(Reviewer #1 Comment 3; Reviewer #3 Comment 7; Reviewer #4 Comment 6), which
flagged the absence of significance testing, confidence intervals, variance
analysis, and ablation studies in the original submission.

Provides:
    - run_multi_seed_evaluation(): repeats the full pipeline (generate -> score ->
      select -> evaluate) over many independent seeds and returns per-run metrics
      for every method.
    - summarize_with_ci(): mean, std, and 95% bootstrap confidence interval per
      method/metric.
    - wilcoxon_vs_reference(): paired Wilcoxon signed-rank test of HASMO against
      every baseline, per metric, with a star-coded significance table.
    - ablation_study(): removes one HASMO objective term at a time (renormalizing
      the remaining weights) and reports the resulting performance drop.
"""

import numpy as np
import pandas as pd
from scipy import stats

from .data_generation import generate_simulation_instance
from .hasmo import hasmo_score, greedy_topk_selection, DEFAULT_WEIGHTS
from .baselines import all_methods
from .evaluate import compute_map_ndcg, precision_recall_per_user


def _method_metrics_for_run(users, interactions, method_scores: dict, top_k=5):
    """Computes ranking metrics (Precision@5, MAP, nDCG@5) and outcome metrics
    (mean achieved satisfaction, mean revenue) for every method on one simulated instance."""
    rows = []
    for name, scores in method_scores.items():
        ranked = interactions.assign(Predicted_Score=scores)
        # Ranking metrics computed over the full candidate list per user (consistent
        # with the manuscript's original evaluation protocol).
        pr, pr_summary = precision_recall_per_user(
            ranked.rename(columns={"True_Relevance": "True_Relevance"}), Ks=[top_k]
        )
        _, map_summary = compute_map_ndcg(ranked)

        sel = greedy_topk_selection(ranked, users, "Predicted_Score", top_k=top_k)
        mean_sat = sel["true_satisfaction"].mean() if len(sel) else np.nan
        mean_rev = sel["revenue"].mean() if len(sel) else np.nan

        prec_row = pr_summary[pr_summary["K"] == top_k]
        rows.append({
            "Method": name,
            f"Precision@{top_k}": float(prec_row["Precision_mean"].iloc[0]) if len(prec_row) else np.nan,
            f"Recall@{top_k}": float(prec_row["Recall_mean"].iloc[0]) if len(prec_row) else np.nan,
            "MAP": map_summary["MAP"],
            f"nDCG@{top_k}": map_summary[f"mean_nDCG@{top_k}"] if f"mean_nDCG@{top_k}" in map_summary else map_summary["mean_nDCG@5"],
            "Mean_Satisfaction": mean_sat,
            "Mean_Revenue": mean_rev,
        })
    return pd.DataFrame(rows)


def run_multi_seed_evaluation(n_users=120, n_pois=40, n_runs=30, top_k=5, base_seed=0):
    """Repeats the pipeline for n_runs independent seeds; returns a long DataFrame
    with one row per (run, method) containing every metric."""
    all_rows = []
    for run in range(n_runs):
        seed = base_seed + run
        users, pois, interactions = generate_simulation_instance(n_users, n_pois, seed)
        method_scores = all_methods(interactions, users, seed=seed)
        run_df = _method_metrics_for_run(users, interactions, method_scores, top_k=top_k)
        run_df["Run"] = run
        all_rows.append(run_df)
    return pd.concat(all_rows, ignore_index=True)


def bootstrap_ci(values: np.ndarray, n_boot=2000, ci=95, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    boot_means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return (lo, hi)


def summarize_with_ci(multi_run_df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    rows = []
    for method, g in multi_run_df.groupby("Method"):
        row = {"Method": method}
        for m in metrics:
            vals = g[m].values
            lo, hi = bootstrap_ci(vals)
            row[f"{m}_mean"] = np.nanmean(vals)
            row[f"{m}_std"] = np.nanstd(vals)
            row[f"{m}_CI95_lo"] = lo
            row[f"{m}_CI95_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def wilcoxon_vs_reference(multi_run_df: pd.DataFrame, metrics: list, reference="IMOHAG (HASMO)"):
    """Paired Wilcoxon signed-rank test of the reference method against every other
    method, matched by Run, for each metric. Returns p-values and significance stars."""
    results = []
    ref = multi_run_df[multi_run_df["Method"] == reference].set_index("Run")
    for method in multi_run_df["Method"].unique():
        if method == reference:
            continue
        other = multi_run_df[multi_run_df["Method"] == method].set_index("Run")
        common_runs = ref.index.intersection(other.index)
        row = {"Comparison": f"{reference} vs {method}"}
        for m in metrics:
            a = ref.loc[common_runs, m].values
            b = other.loc[common_runs, m].values
            mask = ~np.isnan(a) & ~np.isnan(b)
            if mask.sum() < 5 or np.allclose(a[mask], b[mask]):
                row[f"{m}_p"] = np.nan
                row[f"{m}_sig"] = ""
                continue
            try:
                stat_, p = stats.wilcoxon(a[mask], b[mask])
            except ValueError:
                p = np.nan
            row[f"{m}_p"] = p
            if np.isnan(p):
                sig = ""
            elif p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            else:
                sig = "n.s."
            row[f"{m}_sig"] = sig
        results.append(row)
    return pd.DataFrame(results)


def ablation_study(n_users=120, n_pois=40, n_runs=15, top_k=5, base_seed=0):
    """
    Removes one HASMO objective term at a time (weight -> 0, remaining three
    renormalized proportionally) and reports the resulting performance change,
    averaged over n_runs seeds, directly addressing Reviewer #4 Comment 6.
    """
    configs = {
        "Full HASMO": DEFAULT_WEIGHTS,
        "No Relevance term": {"rel": 0.0, "sat": 0.35 / 0.65, "rev": 0.20 / 0.65, "cost": 0.10 / 0.65},
        "No Satisfaction term": {"rel": 0.35 / 0.65, "sat": 0.0, "rev": 0.20 / 0.65, "cost": 0.10 / 0.65},
        "No Revenue term": {"rel": 0.35 / 0.80, "sat": 0.35 / 0.80, "rev": 0.0, "cost": 0.10 / 0.80},
        "No Cost term": {"rel": 0.35 / 0.90, "sat": 0.35 / 0.90, "rev": 0.20 / 0.90, "cost": 0.0},
    }
    rows = []
    for run in range(n_runs):
        seed = base_seed + run
        users, pois, interactions = generate_simulation_instance(n_users, n_pois, seed)
        for cfg_name, w in configs.items():
            scores = hasmo_score(interactions, w)
            sel = greedy_topk_selection(interactions.assign(_score=scores), users, "_score", top_k=top_k)
            rows.append({
                "Run": run, "Configuration": cfg_name,
                "Mean_Relevance": sel["true_relevance_prob"].mean() if len(sel) else np.nan,
                "Mean_Satisfaction": sel["true_satisfaction"].mean() if len(sel) else np.nan,
                "Mean_Revenue": sel["revenue"].mean() if len(sel) else np.nan,
                "Mean_Cost": sel["cost"].mean() if len(sel) else np.nan,
            })
    df = pd.DataFrame(rows)
    summary = df.groupby("Configuration")[["Mean_Relevance", "Mean_Satisfaction", "Mean_Revenue", "Mean_Cost"]].mean().reset_index()
    return df, summary
