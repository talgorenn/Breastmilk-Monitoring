"""
Feature extraction from impedance_differences CSVs.

Two modes:
  cole_cole  - Cole-Cole fitted parameters + spectral summaries + raw ΔZ spectrum
  raw        - full Z1/Z2/ΔZ spectra (real+imag at 30 freqs × 4 pairs = 720 features)

For each (rep, time-step pair, electrode pair):
  cole_cole: fit Cole-Cole to Z1 and Z2, compute deltas, add raw ΔZ spectrum
  raw: return Z1_real/imag, Z2_real/imag, deltaZ_real/imag at all 30 frequencies

Then merge all 4 electrode pairs into one row per (rep, t_i, t_j).
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from cole_cole import fit_cole_cole

DATA_ROOT = Path("C:/Users/tal-m/OneDrive - mail.tau.ac.il/Documents/Final Project/data/data")

ELECTRODE_PAIRS = [
    "electrode1-electrode2",
    "electrode1-electrode3",
    "electrode1-electrode4",
    "electrode3-electrode4",
]

FREQS = [
    0.10, 0.15, 0.22, 0.33, 0.49, 0.73, 1.08, 1.61, 2.40, 3.56,
    5.30, 7.88, 11.72, 17.43, 25.93, 38.57, 57.36, 85.32, 126.90,
    188.74, 280.72, 417.53, 621.02, 923.67, 1373.82, 2043.36,
    3039.20, 4520.35, 6723.36, 10000.00,
]
FREQS = np.array(FREQS)


def _freq_str(f: float) -> str:
    return f"{f:.2f}"


def _extract_spectrum(row: pd.Series, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract real/imag arrays from a row for Z1 or Z2."""
    real = np.array([row[f"{prefix}_{_freq_str(f)}_real"] for f in FREQS])
    imag = np.array([row[f"{prefix}_{_freq_str(f)}_imag"] for f in FREQS])
    return real, imag


def _spectral_features(z_real: np.ndarray, z_imag: np.ndarray, label: str) -> dict:
    """Derived features beyond Cole-Cole: phase angle, |Z|, low/high ratio."""
    mag   = np.sqrt(z_real**2 + z_imag**2)
    phase = np.degrees(np.arctan2(z_imag, z_real))   # degrees
    # low-freq (first 3) vs high-freq (last 3) magnitude ratio
    lh_ratio = mag[:3].mean() / (mag[-3:].mean() + 1e-9)
    # frequency of peak reactance (most negative imag)
    peak_idx = int(np.argmin(z_imag))
    return {
        f"{label}_mag_low":    mag[:3].mean(),
        f"{label}_mag_high":   mag[-3:].mean(),
        f"{label}_lh_ratio":   lh_ratio,
        f"{label}_phase_low":  phase[:3].mean(),
        f"{label}_phase_high": phase[-3:].mean(),
        f"{label}_peak_freq":  FREQS[peak_idx],
        f"{label}_peak_react": z_imag[peak_idx],
    }


def extract_features_for_pair(df: pd.DataFrame, pair: str, mode: str = "cole_cole") -> pd.DataFrame:
    """
    Given a raw impedance_differences DataFrame for one electrode pair,
    return a DataFrame of features per row.

    mode="cole_cole": Cole-Cole params + spectral summaries + raw ΔZ spectrum
    mode="raw":       Z1/Z2/ΔZ real+imag at all 30 frequencies
    """
    prefix = pair.replace("-", "_")
    records = []
    for _, row in df.iterrows():
        r1_real, r1_imag = _extract_spectrum(row, "Z1")
        r2_real, r2_imag = _extract_spectrum(row, "Z2")
        dz_real = np.array([row[f"deltaZ_{_freq_str(f)}_real"] for f in FREQS])
        dz_imag = np.array([row[f"deltaZ_{_freq_str(f)}_imag"] for f in FREQS])

        if mode == "raw":
            feat = {}
            for i, f in enumerate(FREQS):
                fs = _freq_str(f)
                feat[f"{prefix}__Z1_{fs}_real"] = r1_real[i]
                feat[f"{prefix}__Z1_{fs}_imag"] = r1_imag[i]
                feat[f"{prefix}__Z2_{fs}_real"] = r2_real[i]
                feat[f"{prefix}__Z2_{fs}_imag"] = r2_imag[i]
                feat[f"{prefix}__dZ_{fs}_real"] = dz_real[i]
                feat[f"{prefix}__dZ_{fs}_imag"] = dz_imag[i]
        else:
            cc1 = fit_cole_cole(FREQS, r1_real, r1_imag)
            cc2 = fit_cole_cole(FREQS, r2_real, r2_imag)

            feat = {}
            for k, v in cc1.items():
                feat[f"{prefix}__Z1_{k}"] = v
            for k, v in cc2.items():
                feat[f"{prefix}__Z2_{k}"] = v
            for k in ["R0", "R_inf", "fc", "alpha"]:
                feat[f"{prefix}__dcc_{k}"] = cc2[k] - cc1[k]

            feat.update({f"{prefix}__{k}": v for k, v in _spectral_features(r1_real, r1_imag, "Z1").items()})
            feat.update({f"{prefix}__{k}": v for k, v in _spectral_features(r2_real, r2_imag, "Z2").items()})

            for sub in ["mag_low", "mag_high", "lh_ratio", "phase_low", "phase_high"]:
                feat[f"{prefix}__d_{sub}"] = feat[f"{prefix}__Z2_{sub}"] - feat[f"{prefix}__Z1_{sub}"]

            # raw ΔZ spectrum alongside Cole-Cole features
            for i, f in enumerate(FREQS):
                fs = _freq_str(f)
                feat[f"{prefix}__dZ_{fs}_real"] = dz_real[i]
                feat[f"{prefix}__dZ_{fs}_imag"] = dz_imag[i]

        records.append(feat)

    return pd.DataFrame(records)


_FEATURE_MODE = "cole_cole"  # used only in single-process paths


def _load_rep_with_mode(args):
    rep_dir, mode = args
    return load_rep_features(rep_dir, mode=mode)


def load_rep_features(rep_dir: Path, mode: str = "cole_cole") -> pd.DataFrame | None:
    """Load one rep and return physics-feature rows."""
    name = rep_dir.name
    m = re.match(r"rep_(\d+)_gauss_([\d.]+)a_s(\d+)", name)
    if not m:
        return None

    rep_id = int(m.group(1))
    gauss  = float(m.group(2))
    seed   = int(m.group(3))

    pair_frames = []
    for pair in ELECTRODE_PAIRS:
        csv = rep_dir / pair / "impedance_differences.csv"
        if not csv.exists():
            return None
        df_raw = pd.read_csv(csv)
        feat_df = extract_features_for_pair(df_raw, pair, mode=mode)
        pair_frames.append(feat_df.reset_index(drop=True))

    # meta columns from first pair
    first = pd.read_csv(rep_dir / ELECTRODE_PAIRS[0] / "impedance_differences.csv")
    meta = first[["t_i", "t_j", "dt_steps", "fluid_diff_ml"]].copy()
    meta["rep_id"] = rep_id
    meta["gauss"]  = gauss
    meta["seed"]   = seed

    combined = pd.concat([meta.reset_index(drop=True)] + pair_frames, axis=1)
    return combined


def load_sample_features(n_reps: int = 200, seed: int = 0, mode: str = "cole_cole") -> pd.DataFrame:
    rep_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
    rng = np.random.default_rng(seed)
    chosen = rng.choice(rep_dirs, size=min(n_reps, len(rep_dirs)), replace=False)

    frames = []
    for i, d in enumerate(chosen):
        df = load_rep_features(d, mode=mode)
        if df is not None:
            frames.append(df)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(chosen)} reps processed")

    return pd.concat(frames, ignore_index=True)


def load_all_features(n_workers: int = 4, mode: str = "cole_cole") -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    rep_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])

    frames = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_load_rep_with_mode, (d, mode)): d for d in rep_dirs}
        for i, fut in enumerate(as_completed(futures)):
            df = fut.result()
            if df is not None:
                frames.append(df)
            if (i + 1) % 500 == 0:
                print(f"  loaded {i+1}/{len(rep_dirs)} reps")

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    print("Testing Cole-Cole feature extraction on 10 reps...")
    df = load_sample_features(10)
    print(f"Shape: {df.shape}")
    print(f"Feature columns: {[c for c in df.columns if '__' in c][:10]} ...")
    print(df[["fluid_diff_ml", "dt_steps"]].describe())
