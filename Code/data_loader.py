"""
Loads bioimpedance simulation data and combines all 4 electrode pairs
into a single feature row per (rep, t_i, t_j) observation.
"""
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

_CANDIDATE_ROOTS = [
    Path("C:/Users/tal-m/OneDrive - mail.tau.ac.il/Documents/Final Project/data/data"),
    Path(__file__).parent.parent / "data" / "data",
    Path("data/data"),
]


def _find_data_root() -> Path:
    for p in _CANDIDATE_ROOTS:
        if p.exists() and any(p.iterdir()):
            return p
    raise FileNotFoundError(
        "Could not find data directory. Tried:\n"
        + "\n".join(f"  {p}" for p in _CANDIDATE_ROOTS)
        + "\nExtract data.zip or update _CANDIDATE_ROOTS in data_loader.py."
    )


DATA_ROOT = _find_data_root()

ELECTRODE_PAIRS = [
    "electrode1-electrode2",
    "electrode1-electrode3",
    "electrode1-electrode4",
    "electrode3-electrode4",
]

IMPEDANCE_COLS = None  # will be inferred from first file


def _impedance_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"fluid_t", "fluid_t+b", "fluid_diff", "fluid_t_ml", "fluid_t+b_ml",
               "fluid_diff_ml", "t_i", "t_j", "dt_steps"}
    return [c for c in df.columns if c not in exclude]


def load_rep(rep_dir: Path) -> pd.DataFrame | None:
    """Load one rep: concatenate impedance features from all electrode pairs horizontally."""
    dfs = []
    for pair in ELECTRODE_PAIRS:
        csv = rep_dir / pair / "impedance_differences.csv"
        if not csv.exists():
            return None
        df = pd.read_csv(csv)
        imp_cols = _impedance_cols(df)
        # prefix columns with electrode pair to avoid name collisions
        prefix = pair.replace("-", "_")
        renamed = df[imp_cols].rename(columns={c: f"{prefix}__{c}" for c in imp_cols})
        dfs.append(renamed)

    # all pairs share the same meta columns — take from first pair
    first = pd.read_csv(rep_dir / ELECTRODE_PAIRS[0] / "impedance_differences.csv")
    meta = first[["t_i", "t_j", "dt_steps", "fluid_diff_ml"]].copy()

    # parse rep metadata from folder name
    name = rep_dir.name
    m = re.match(r"rep_(\d+)_gauss_([\d.]+)a_s(\d+)", name)
    if m:
        meta["rep_id"] = int(m.group(1))
        meta["gauss"]  = float(m.group(2))
        meta["seed"]   = int(m.group(3))
    else:
        return None  # skip non-matching dirs (e.g. .zip extracted folders)

    combined = pd.concat([meta.reset_index(drop=True)] + [d.reset_index(drop=True) for d in dfs], axis=1)
    return combined


def load_sample(n_reps: int = 200, seed: int = 0) -> pd.DataFrame:
    """Load a random sample of n_reps replicates."""
    rep_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
    rng = np.random.default_rng(seed)
    chosen = rng.choice(rep_dirs, size=min(n_reps, len(rep_dirs)), replace=False)

    frames = []
    for d in chosen:
        df = load_rep(d)
        if df is not None:
            frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_all(n_workers: int = 4) -> pd.DataFrame:
    """Load all replicates."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    rep_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])

    frames = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(load_rep, d): d for d in rep_dirs}
        for i, fut in enumerate(as_completed(futures)):
            df = fut.result()
            if df is not None:
                frames.append(df)
            if (i + 1) % 500 == 0:
                print(f"  loaded {i+1}/{len(rep_dirs)} reps")

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    print("Loading 200-rep sample...")
    df = load_sample(200)
    print(f"Shape: {df.shape}")
    print(f"Target stats:\n{df['fluid_diff_ml'].describe()}")
    print(f"Features (first 5): {[c for c in df.columns if '__' in c][:5]}")
