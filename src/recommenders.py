import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class ItemItemCF:
    """Simple item-item collaborative filtering using co-occurrence (binary relevance) and cosine similarity."""
    def __init__(self, interactions_df):
        # interactions_df: columns ['User_ID','POI_ID','True_Relevance']
        self.interactions = interactions_df.copy()
        self.user_item = self.interactions.pivot_table(index='User_ID', columns='POI_ID', values='True_Relevance', fill_value=0)
        # item-item similarity
        self.item_ids = list(self.user_item.columns)
        self.sim = cosine_similarity(self.user_item.T)
        # map id -> index
        self.idx_map = {pid:i for i,pid in enumerate(self.item_ids)}

    def score_user(self, user_id):
        if user_id not in self.user_item.index:
            # cold user: return zeros
            return pd.Series(0, index=self.item_ids)
        user_vec = self.user_item.loc[user_id].values
        scores = self.sim.dot(user_vec)
        return pd.Series(scores, index=self.item_ids)

    def recommend_for_user(self, user_id, topk=10):
        scores = self.score_user(user_id)
        return scores.sort_values(ascending=False).head(topk)

class ContentBased:
    """Content-based recommender using TF-IDF on an item text field (e.g., Visited_Sites)."""
    def __init__(self, items_df, text_col='Visited_Sites'):
        # items_df must contain 'POI_ID' and text_col
        self.items = items_df.copy()
        if text_col in items_df.columns:
            corpus = items_df[text_col].astype(str).values
            self.vec = TfidfVectorizer(max_features=500)
            self.feats = self.vec.fit_transform(corpus)
            self.index = items_df['POI_ID'].values
        else:
            # fallback: one-hot on POI_ID
            self.index = items_df['POI_ID'].values
            from scipy.sparse import csr_matrix
            self.feats = csr_matrix(np.eye(len(self.index)))

    def score_user_profile(self, user_profile_vector):
        # user_profile_vector should be same dimension as features columns
        scores = self.feats.dot(user_profile_vector)
        return pd.Series(scores.flatten(), index=self.index)

    def score_user_from_history(self, user_history_poi_ids):
        # simple profile: mean of item vectors in history
        mask = np.isin(self.index, user_history_poi_ids)
        if mask.sum() == 0:
            return pd.Series(0, index=self.index)
        user_vec = self.feats[mask].mean(axis=0)
        scores = self.feats.dot(user_vec.T)
        return pd.Series(scores.flatten(), index=self.index)

class HybridRecommender:
    """Weighted hybrid of ItemItemCF and ContentBased."""
    def __init__(self, cf: ItemItemCF, cbf: ContentBased, alpha=0.6):
        self.cf = cf
        self.cbf = cbf
        self.alpha = alpha

    def score_user(self, user_id, user_history=None):
        cf_scores = self.cf.score_user(user_id)
        if user_history is not None:
            cbf_scores = self.cbf.score_user_from_history(user_history)
        else:
            cbf_scores = pd.Series(0, index=cf_scores.index)
        # align indices and combine
        common = sorted(list(set(cf_scores.index).union(set(cbf_scores.index))))
        cf_v = cf_scores.reindex(common).fillna(0)
        cbf_v = cbf_scores.reindex(common).fillna(0)
        hybrid = self.alpha * cf_v + (1 - self.alpha) * cbf_v
        return hybrid.sort_values(ascending=False)

    def recommend(self, user_id, user_history=None, topk=10):
        return self.score_user(user_id, user_history=user_history).head(topk)

if __name__ == '__main__':
    print('Recommenders module loaded')
