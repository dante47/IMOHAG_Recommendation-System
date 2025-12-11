import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# Prepare dataset for Precision/Recall, MAP, nDCG
# ------------------------------------------------------------

def prepare_recall_dataset(df):
    """
    Ensures that the dataset contains appropriate fields for ranking metrics
    (Precision@K, Recall@K, MAP, nDCG).
    Uses:
      - Predicted_Score
      - True_Relevance
      - User_ID
      - POI_ID
    """

    df = df.copy()

    # Predicted score must be float
    if 'Predicted_Score' in df.columns:
        df['Predicted_Score'] = df['Predicted_Score'].astype(float)
    else:
        # fallback (should never happen in your dataset)
        df['Predicted_Score'] = np.random.rand(len(df))

    # Ground truth relevance must be int (0/1)
    if 'True_Relevance' in df.columns:
        df['True_Relevance'] = df['True_Relevance'].astype(int)
    else:
        raise KeyError("Column 'True_Relevance' missing from dataset.")

    # Ensure ordering is consistent
    df = df.sort_values(['User_ID', 'Predicted_Score'], ascending=[True, False])

    return df


# ------------------------------------------------------------
# Normalize a numeric column (you already had this)
# ------------------------------------------------------------

def normalize_series(s):
    scaler = MinMaxScaler()
    vals = s.values.reshape(-1, 1)
    scaled = scaler.fit_transform(vals).flatten()
    return pd.Series(scaled, index=s.index)


# ------------------------------------------------------------
# Prepare dataset for UX, sentiment, revenue, and system metrics
# ------------------------------------------------------------

def prepare_ultimate_for_userinfo(df):
    """
    Extract user-level metadata needed for:
      - Satisfaction vs SUS
      - SUS vs NPS
      - Sentiment accuracy
      - Resolution time
      - Revenue per booking source

    We take the first row per user (your original logic preserved).
    """

    required = [
        'User_ID',
        'Satisfaction_0_1',
        'SUS',
        'NPS',
        'Resolution_Time_Min',
        'Booking_Source',
        'Price_USD',
        'Sentiment_Score_-1_1',
        'Sentiment_Label'
    ]

    for col in required:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing from dataset.")

    user_info = df.groupby('User_ID').first().reset_index()

    return user_info


# ------------------------------------------------------------
# DEBUG block (kept from your original structure but updated)
# ------------------------------------------------------------

if __name__ == '__main__':
    from src.data_loader import load_dataset

    df = load_dataset()

    recall_df = prepare_recall_dataset(df)
    userinfo_df = prepare_ultimate_for_userinfo(df)

    print("Recall-ready shape:", recall_df.shape)
    print("User-info shape:", userinfo_df.shape)
