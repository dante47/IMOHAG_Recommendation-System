import pandas as pd
import os

ULT_PATH = '/mnt/data/Djanet_Hybrid_Tourism_Dataset_ULTIMATE.xlsx'
RECALL_READY_PATH = '/mnt/data/Djanet_Hybrid_Tourism_Dataset_RECALL_READY.xlsx'

def load_ultimate(path=ULT_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ultimate dataset not found at {path}")
    return pd.read_excel(path)

def load_recall_ready(path=RECALL_READY_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Recall-ready dataset not found at {path}")
    return pd.read_excel(path)

if __name__ == '__main__':
    print('Ultimate loaded rows:', load_ultimate().shape[0])
    print('Recall-ready loaded rows:', load_recall_ready().shape[0])
