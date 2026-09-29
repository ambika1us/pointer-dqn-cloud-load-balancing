# src/data_loader.py
import pandas as pd

from config import (
    GOOGLE_TRACE_PATH,
    ALIBABA_CONTAINER_META,
    ALIBABA_MACHINE_META,
    ALIBABA_CONTAINER_USAGE,
    ALIBABA_CONTAINER_META_COLS,
    ALIBABA_MACHINE_META_COLS,
)


def load_google_trace(path=GOOGLE_TRACE_PATH) -> pd.DataFrame:
    """Load the Google ClusterData 2019 (Borg) trace."""
    df = pd.read_csv(path, low_memory=False)
    print(f"[data_loader] Loaded Google trace: shape={df.shape}")
    print(f"[data_loader] Columns: {df.columns.tolist()}")
    print(f"[data_loader] event values: {df['event'].dropna().unique()[:10].tolist()}")
    return df


def load_alibaba_container_meta(path=ALIBABA_CONTAINER_META) -> pd.DataFrame:
    """Task source: header-less 8-column file with container requests."""
    df = pd.read_csv(
        path,
        header=None,
        names=ALIBABA_CONTAINER_META_COLS,
        low_memory=False,
    )
    print(f"[data_loader] container_meta: shape={df.shape}")
    print(f"[data_loader] Columns: {df.columns.tolist()}")
    print(f"[data_loader] status values: {df['status'].unique()[:5].tolist()}")
    return df


def load_alibaba_machine_meta(path=ALIBABA_MACHINE_META) -> pd.DataFrame:
    """Machine capacity: header-less 7-column file."""
    df = pd.read_csv(
        path,
        header=None,
        names=ALIBABA_MACHINE_META_COLS,
        low_memory=False,
    )
    print(f"[data_loader] machine_meta: shape={df.shape}")
    print(f"[data_loader] Columns: {df.columns.tolist()}")
    return df


def load_alibaba_container_usage(path=ALIBABA_CONTAINER_USAGE) -> pd.DataFrame:
    """Real per-container usage: has header row."""
    df = pd.read_csv(path, header=0, low_memory=False)
    print(f"[data_loader] container_usage: shape={df.shape}")
    print(f"[data_loader] Columns: {df.columns.tolist()}")
    return df