import pandas as pd
import os

# NOTE: the original hardcoded absolute paths (/mnt/data/...) only existed on the
# authors' local machine and are not portable/reproducible on a clean checkout.
# `load_dataset()` below resolves the legacy static spreadsheet relative to the
# repository root instead, and falls back clearly if it is not found. For a fully
# reproducible, parameterized alternative that does not depend on a static file at
# all, use `src.data_generation.generate_simulation_instance` (see README, "Revision
# Update" section, added in response to peer review).

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_LEGACY_PATH = os.path.join(_REPO_ROOT, "Datasets + Results", "Based_Djanet_Dataset.xlsx")
ULT_PATH = os.environ.get("IMOHAG_ULTIMATE_PATH", DEFAULT_LEGACY_PATH)
RECALL_READY_PATH = os.environ.get("IMOHAG_RECALL_READY_PATH", DEFAULT_LEGACY_PATH)

def load_ultimate(path=None):
    path = path or ULT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ultimate dataset not found at {path}")
    return pd.read_excel(path)

def load_recall_ready(path=None):
    path = path or RECALL_READY_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Recall-ready dataset not found at {path}")
    return pd.read_excel(path)

def load_dataset(path=None):
    """Compatibility alias used by scripts/run_all.py."""
    return load_ultimate(path)

if __name__ == '__main__':
    print('Ultimate loaded rows:', load_ultimate().shape[0])
    print('Recall-ready loaded rows:', load_recall_ready().shape[0])
