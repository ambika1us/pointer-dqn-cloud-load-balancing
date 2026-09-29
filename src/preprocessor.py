# src/preprocessor.py
import ast
import numpy as np
import pandas as pd

from config import PROCESSED_GOOGLE, PROCESSED_ALIBABA, PROCESSED_DIR

# src/preprocessor.py — add at top

def _sample_durations_from_google(n: int, seed: int = 42) -> np.ndarray:
    """Sample n durations from the Google trace's empirical distribution."""
    google = pd.read_csv(PROCESSED_GOOGLE)
    rng = np.random.default_rng(seed)
    return rng.choice(google["duration"].values, size=n, replace=True)

def _parse_nested(x) -> dict:
    """Parse a dict-like string into a dict. Returns {} on failure."""
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return {}
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return {}


def preprocess_google(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print(f"[google-pre] start rows={len(df)}")

    # ---- 1. Keep SCHEDULE and FINISH events ----
    df = df[df["event"].isin(["SCHEDULE", "FINISH"])].copy()
    print(f"[google-pre] after event filter rows={len(df)}")

    # ---- 2. Time (microseconds -> seconds) ----
    df["time"] = pd.to_numeric(df["time"], errors="coerce") / 1e6
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_numeric(df["end_time"], errors="coerce")

    # ---- 3. Duration ----
    df["duration"] = (df["end_time"] - df["start_time"]) / 1e6
    df["duration"] = df["duration"].fillna(0.0)

    # If most durations are zero, fall back to FINISH-only rows
    if (df["duration"] <= 0).mean() > 0.5:
        print("[google-pre] Most durations invalid; keeping FINISH rows only")
        df = df[df["event"] == "FINISH"].copy()
        df["duration"] = (df["end_time"] - df["start_time"]) / 1e6

    df["duration"] = df["duration"].clip(lower=0.001, upper=3600.0)
    sub_300 = df[df["duration"] < 300.0]["duration"].values
    if len(sub_300) > 0 and (df["duration"] == 300.0).mean() > 0.3:
        print(f"[google-pre] 300s artifact detected — resampling from {len(sub_300)} sub-300 durations")
    rng = np.random.default_rng(42)
    df["duration"] = rng.choice(sub_300, size=len(df), replace=True)

    print(f"[google-pre] duration NaN={df['duration'].isna().sum()}  "
      f"range=[{df['duration'].min():.4f}, {df['duration'].max():.4f}]  "
      f"median={df['duration'].median():.4f}")
    print(f"[google-pre] duration NaN={df['duration'].isna().sum()}  "
          f"range=[{df['duration'].min():.4f}, {df['duration'].max():.4f}]")

    # ---- 4. Parse resource_request (primary source) ----
    req = df["resource_request"].apply(_parse_nested)
    df["cpu"] = pd.to_numeric(
        req.apply(lambda d: d.get("cpus", np.nan)), errors="coerce"
    )
    df["memory"] = pd.to_numeric(
        req.apply(lambda d: d.get("memory", np.nan)), errors="coerce"
    )
    print(f"[google-pre] cpu NaN after resource_request={df['cpu'].isna().sum()}")
    print(f"[google-pre] memory NaN after resource_request={df['memory'].isna().sum()}")

    # ---- 5. Fallback to average_usage / maximum_usage if needed ----
    if df["cpu"].isna().mean() > 0.5:
        print("[google-pre] Falling back to average_usage/maximum_usage")
        avg = df["average_usage"].apply(_parse_nested)
        maxu = df["maximum_usage"].apply(_parse_nested)
        cpu_avg = pd.to_numeric(avg.apply(lambda d: d.get("cpus", np.nan)), errors="coerce")
        cpu_max = pd.to_numeric(maxu.apply(lambda d: d.get("cpus", np.nan)), errors="coerce")
        mem_avg = pd.to_numeric(avg.apply(lambda d: d.get("memory", np.nan)), errors="coerce")
        mem_max = pd.to_numeric(maxu.apply(lambda d: d.get("memory", np.nan)), errors="coerce")
        df["cpu"] = df["cpu"].fillna(cpu_avg).fillna(cpu_max)
        df["memory"] = df["memory"].fillna(mem_avg).fillna(mem_max)

    # ---- 6. Final NaN fill via median ----
    if df["cpu"].isna().all() or df["memory"].isna().all():
        raise ValueError("CPU/memory entirely NaN after all fallbacks.")

    df["cpu"] = df["cpu"].fillna(df["cpu"].median())
    df["memory"] = df["memory"].fillna(df["memory"].median())

    # ---- 7. Trace-invariant normalization ----
    df["cpu"] = df["cpu"] / df["cpu"].max()
    df["memory"] = df["memory"] / df["memory"].max()

    # ---- 8. Drop rows with NaN in required columns ----
    df = df.dropna(subset=["time", "cpu", "memory", "duration"])
    print(f"[google-pre] after dropna rows={len(df)}")

    if len(df) == 0:
        raise ValueError("No valid rows after preprocessing.")

    # ---- 9. Sort and finalize ----
    df = df.sort_values("time").reset_index(drop=True)
    df = df[["time", "machine_id", "cpu", "memory", "duration"]]

    print(f"[google-pre] final rows={len(df)}")
    print(f"[google-pre] CPU range=[{df['cpu'].min():.4f}, {df['cpu'].max():.4f}]")
    print(f"[google-pre] Memory range=[{df['memory'].min():.4f}, {df['memory'].max():.4f}]")
    print(f"[google-pre] Duration range=[{df['duration'].min():.4f}, {df['duration'].max():.4f}]")
    return df

def preprocess_alibaba(container_meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert container_meta into schema:
        time, machine_id, cpu, memory, duration
    """
    df = container_meta_df.copy()
    print(f"[alibaba-pre] start rows={len(df)}")

    # 1. Keep only 'started' containers
    if "status" in df.columns:
        before = len(df)
        df = df[df["status"].astype(str).str.lower() == "started"].copy()
        print(f"[alibaba-pre] after status filter rows={len(df)} (was {before})")

    # 2. Numeric conversion for time and resources only
    #    (machine_id stays as string — Alibaba uses IDs like 'm_2556')
    for col in ["time_stamp", "cpu_request", "mem_request"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["time_stamp", "cpu_request", "mem_request"])
    print(f"[alibaba-pre] after numeric dropna rows={len(df)} (was {before})")

    # 3. Convert machine_id to categorical integer (keeps numeric column)
    df["machine_id"] = df["machine_id"].astype("category").cat.codes
    print(f"[alibaba-pre] unique machines: {df['machine_id'].nunique()}")

    # 4. Drop zero-resource rows
    before = len(df)
    df = df[(df["cpu_request"] > 0) & (df["mem_request"] > 0)].copy()
    print(f"[alibaba-pre] after zero-resource filter rows={len(df)} (was {before})")

    # 5. Time reference (time_stamp in seconds)
    df["time"] = df["time_stamp"] - df["time_stamp"].min()

    # 6. Synthetic duration (constant 60 sec)
    df["duration"] = _sample_durations_from_google(len(df))

    # 7. Trace-invariant normalization
    df["cpu"] = df["cpu_request"] / df["cpu_request"].max()
    df["memory"] = df["mem_request"] / df["mem_request"].max()

    # 8. Sort by arrival
    df = df.sort_values("time").reset_index(drop=True)

    # 9. Subsample to ~160k rows for comparability with Google
    if len(df) > 200_000:
        step = len(df) // 160_000
        df = df.iloc[::step].reset_index(drop=True)
        print(f"[alibaba-pre] subsampled to {len(df)} rows")

    # 10. Final schema
    df = df[["time", "machine_id", "cpu", "memory", "duration"]]

    print(f"[alibaba-pre] final rows={len(df)}")
    print(f"[alibaba-pre] CPU range=[{df['cpu'].min():.4f}, {df['cpu'].max():.4f}]")
    print(f"[alibaba-pre] Memory range=[{df['memory'].min():.4f}, {df['memory'].max():.4f}]")
    print(f"[alibaba-pre] Time range=[{df['time'].min():.2f}, {df['time'].max():.2f}]")
    return df

def main():
    import sys
    trace = sys.argv[1] if len(sys.argv) > 1 else "google"

    if trace == "google":
        from src.data_loader import load_google_trace
        print("=" * 60)
        print("Preprocessing GOOGLE trace")
        print("=" * 60)
        df = load_google_trace()
        df = preprocess_google(df)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_GOOGLE, index=False)
        print(f"Wrote {len(df)} rows -> {PROCESSED_GOOGLE}")

    elif trace == "alibaba":
        from src.data_loader import load_alibaba_container_meta
        print("=" * 60)
        print("Preprocessing ALIBABA trace")
        print("=" * 60)
        df = load_alibaba_container_meta()
        df = preprocess_alibaba(df)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_ALIBABA, index=False)
        print(f"Wrote {len(df)} rows -> {PROCESSED_ALIBABA}")

    else:
        raise ValueError(f"Unknown trace: {trace}. Use 'google' or 'alibaba'.")


if __name__ == "__main__":
    main()