import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_recall_dataset(df):
    # Ensure expected columns exist and proper dtypes
    df = df.copy()
    if 'Predicted_Score' in df.columns:
        df['Predicted_Score'] = df['Predicted_Score'].astype(float)
    else:
        df['Predicted_Score'] = np.random.rand(len(df))
    df['True_Relevance'] = df['True_Relevance'].astype(int)
    return df

def normalize_series(s):
    scaler = MinMaxScaler()
    vals = s.values.reshape(-1,1)
    scaled = scaler.fit_transform(vals).flatten()
    return pd.Series(scaled, index=s.index)

def prepare_ultimate_for_userinfo(df):
    # take first row per user as user info
    user_info = df.groupby('User_ID').first().reset_index()
    return user_info

if __name__ == '__main__':
    from .data_loader import load_recall_ready, load_ultimate
    r = load_recall_ready()
    u = load_ultimate()
    print('Prepared sizes', prepare_recall_dataset(r).shape, prepare_ultimate_for_userinfo(u).shape)
