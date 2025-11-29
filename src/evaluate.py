import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(os.getcwd(), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def precision_recall_per_user(recall_df, Ks=[1,3,5,10]):
    rows = []
    users = recall_df['User_ID'].unique()
    for u in users:
        urows = recall_df[recall_df['User_ID'] == u].sort_values('Predicted_Score', ascending=False)
        total_rel = int(urows['True_Relevance'].sum())
        for K in Ks:
            topk = urows.head(K)
            rel_topk = int(topk['True_Relevance'].sum())
            precision = rel_topk / K
            recall = rel_topk / total_rel if total_rel > 0 else np.nan
            rows.append({'User_ID': u, 'K': K, 'Precision': precision, 'Recall': recall, 'Total_Relevant_Items': total_rel})
    pr = pd.DataFrame(rows)
    pr_summary = pr.groupby('K').agg(Precision_mean=('Precision','mean'), Recall_mean=('Recall','mean')).reset_index()
    return pr, pr_summary

def average_precision(rels):
    rels = np.asarray(rels).astype(int)
    if rels.sum() == 0:
        return np.nan
    precisions = []
    rel_count = 0
    for i, r in enumerate(rels, start=1):
        if r:
            rel_count += 1
            precisions.append(rel_count / i)
    return np.mean(precisions) if precisions else np.nan

def dcg_at_k(rels, k):
    rels = np.asarray(rels).astype(int)[:k]
    if rels.size == 0:
        return 0.0
    gains = 2**rels - 1
    discounts = np.log2(np.arange(2, gains.size + 2))
    return float((gains / discounts).sum())

def ndcg_at_k(rels, k):
    rels = np.asarray(rels).astype(int)
    dcg = dcg_at_k(rels, k)
    idcg = dcg_at_k(np.sort(rels)[::-1], k)
    return float(dcg / idcg) if idcg > 0 else 0.0

def compute_map_ndcg(recall_df):
    users = recall_df['User_ID'].unique()
    rows = []
    ap_list = []
    ndcg5 = []
    ndcg10 = []
    for u in users:
        urows = recall_df[recall_df['User_ID'] == u].sort_values('Predicted_Score', ascending=False)
        rels = urows['True_Relevance'].values
        ap = average_precision(rels)
        n5 = ndcg_at_k(rels, 5)
        n10 = ndcg_at_k(rels, 10)
        rows.append({'User_ID': u, 'AP': ap, 'nDCG@5': n5, 'nDCG@10': n10, 'Total_Relevant': int(urows['True_Relevance'].sum())})
        if not np.isnan(ap):
            ap_list.append(ap)
        ndcg5.append(n5)
        ndcg10.append(n10)
    summary = {'MAP': np.nanmean(ap_list) if ap_list else np.nan, 'mean_nDCG@5': np.mean(ndcg5), 'mean_nDCG@10': np.mean(ndcg10)}
    return pd.DataFrame(rows), summary

def plot_pr(summary_df, out_png):
    plt.figure(figsize=(6,4))
    plt.plot(summary_df['Recall_mean'], summary_df['Precision_mean'], marker='o')
    plt.xlabel('Average Recall@K')
    plt.ylabel('Average Precision@K')
    plt.title('Precision@K vs Recall@K')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

if __name__ == '__main__':
    print('Evaluate module ready')
