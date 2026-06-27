"""
1D CNN regression model to predict fluid_diff_ml from raw impedance spectra.

The frequency axis is treated as a 1D signal. For each electrode pair we have
6 channels (Z1_real, Z1_imag, Z2_real, Z2_imag, dZ_real, dZ_imag) × 30 frequencies.
All 4 electrode pairs are stacked → input shape: (batch, 24 channels, 30 freq points).

dt_steps is injected as a scalar after the conv layers.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from features import load_sample_features, load_all_features, ELECTRODE_PAIRS, FREQS

OUTPUT_DIR = Path("C:/Users/tal-m/MyProject/output")
OUTPUT_DIR.mkdir(exist_ok=True)

NON_FEATURE = {"fluid_diff_ml", "t_i", "t_j", "rep_id", "seed",
               "fluid_t", "fluid_t+b", "fluid_diff", "gauss"}

CHANNELS = ["Z1_real", "Z1_imag", "Z2_real", "Z2_imag", "dZ_real", "dZ_imag"]
N_CHANNELS = len(CHANNELS) * len(ELECTRODE_PAIRS)  # 6 × 4 = 24
N_FREQS = len(FREQS)  # 30


def df_to_tensors(df: pd.DataFrame):
    """
    Returns:
      X_spec  : (N, 24, 30)  - spectral input for conv layers
      X_scalar: (N, 1)       - dt_steps
      y       : (N,)         - fluid_diff_ml
    """
    n = len(df)
    X_spec = np.zeros((n, N_CHANNELS, N_FREQS), dtype=np.float32)

    ch = 0
    for pair in ELECTRODE_PAIRS:
        prefix = pair.replace("-", "_")
        for sig, col_tmpl in [
            ("Z1_real", "Z1_{}_real"), ("Z1_imag", "Z1_{}_imag"),
            ("Z2_real", "Z2_{}_real"), ("Z2_imag", "Z2_{}_imag"),
            ("dZ_real", "dZ_{}_real"), ("dZ_imag", "dZ_{}_imag"),
        ]:
            for fi, f in enumerate(FREQS):
                fs = f"{f:.2f}"
                col = f"{prefix}__{col_tmpl.format(fs)}"
                X_spec[:, ch, fi] = df[col].values
            ch += 1

    X_scalar = df[["dt_steps"]].values.astype(np.float32)
    y = df["fluid_diff_ml"].values.astype(np.float32)
    return X_spec, X_scalar, y


def split_by_rep(df, val_frac=0.15, test_frac=0.15, seed=42):
    rep_ids = df["rep_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tv_idx, te_idx = next(gss.split(df, groups=rep_ids))
    df_tv, df_te = df.iloc[tv_idx], df.iloc[te_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac / (1 - test_frac), random_state=seed)
    tr_idx, va_idx = next(gss2.split(df_tv, groups=df_tv["rep_id"].values))
    return df_tv.iloc[tr_idx], df_tv.iloc[va_idx], df_te


class SpectralCNN(nn.Module):
    """
    1D CNN along the frequency axis, with a scalar side-input for dt_steps.
    """
    def __init__(self, n_channels: int = N_CHANNELS, n_freqs: int = N_FREQS,
                 n_scalar: int = 1, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            # (B, 24, 30) → (B, 64, 30)
            nn.Conv1d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            # (B, 64, 30) → (B, 128, 15)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            # (B, 128, 15) → (B, 128, 15)
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),  # → (B, 128, 4)
        )
        conv_out = 128 * 4

        self.head = nn.Sequential(
            nn.Linear(conv_out + n_scalar, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_spec, x_scalar):
        z = self.conv(x_spec)
        z = z.flatten(1)
        z = torch.cat([z, x_scalar], dim=1)
        return self.head(z).squeeze(1)


def train(df: pd.DataFrame, tag: str = "cnn", epochs: int = 150,
          batch_size: int = 256, lr: float = 1e-3):

    print(f"Rows: {len(df)}")
    train_df, val_df, test_df = split_by_rep(df)
    print(f"Split — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # fit scaler on spectral channels using training data only
    Xtr_spec, Xtr_sc, ytr = df_to_tensors(train_df)
    Xva_spec, Xva_sc, yva = df_to_tensors(val_df)
    Xte_spec, Xte_sc, yte = df_to_tensors(test_df)

    # normalise each channel independently across freq points
    ch_mean = Xtr_spec.mean(axis=(0, 2), keepdims=True)   # (1, 24, 1)
    ch_std  = Xtr_spec.std(axis=(0, 2), keepdims=True) + 1e-8
    Xtr_spec = (Xtr_spec - ch_mean) / ch_std
    Xva_spec = (Xva_spec - ch_mean) / ch_std
    Xte_spec = (Xte_spec - ch_mean) / ch_std

    sc_scaler = StandardScaler()
    Xtr_sc = sc_scaler.fit_transform(Xtr_sc)
    Xva_sc = sc_scaler.transform(Xva_sc)
    Xte_sc = sc_scaler.transform(Xte_sc)

    def to_ds(Xs, Xsc, y):
        return TensorDataset(
            torch.tensor(Xs), torch.tensor(Xsc.astype(np.float32)), torch.tensor(y)
        )

    train_loader = DataLoader(to_ds(Xtr_spec, Xtr_sc, ytr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(to_ds(Xva_spec, Xva_sc, yva), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SpectralCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_count = 0
    patience_limit = 25

    for epoch in range(1, epochs + 1):
        model.train()
        for xs, xsc, yb in train_loader:
            xs, xsc, yb = xs.to(device), xsc.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xs, xsc), yb).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xs, xsc, yb in val_loader:
                xs, xsc, yb = xs.to(device), xsc.to(device), yb.to(device)
                val_loss += criterion(model(xs, xsc), yb).item() * len(yb)
        val_loss /= len(val_df)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}  val_loss={val_loss:.2f}  best={best_val_loss:.2f}")

        if patience_count >= patience_limit:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(Xte_spec).to(device)
        Xsc_t = torch.tensor(Xte_sc.astype(np.float32)).to(device)
        preds = model(Xte_t, Xsc_t).cpu().numpy()

    rmse = mean_squared_error(yte, preds) ** 0.5
    mae  = mean_absolute_error(yte, preds)
    r2   = r2_score(yte, preds)
    print(f"\nTest  RMSE: {rmse:.4f} mL  MAE: {mae:.4f} mL  R2: {r2:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yte, preds, alpha=0.3, s=10)
    lims = [min(yte.min(), preds.min()), max(yte.max(), preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual fluid_diff_ml (mL)")
    ax.set_ylabel("Predicted fluid_diff_ml (mL)")
    ax.set_title(f"{tag}  R2={r2:.3f}  RMSE={rmse:.3f} mL")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_pred_vs_actual.png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {OUTPUT_DIR / f'{tag}_pred_vs_actual.png'}")

    torch.save(best_state, OUTPUT_DIR / f"{tag}.pt")
    joblib.dump((ch_mean, ch_std, sc_scaler), OUTPUT_DIR / f"{tag}_scalers.pkl")
    print(f"Model saved to {OUTPUT_DIR / f'{tag}.pt'}")
    return model, {"rmse": rmse, "mae": mae, "r2": r2}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    tag = f"cnn_n{args.sample if args.sample else 'all'}"

    if args.sample > 0:
        print(f"Loading raw features for {args.sample} reps...")
        df = load_sample_features(args.sample, mode="raw")
    else:
        df = load_all_features(mode="raw")

    print(f"Loaded {len(df)} rows from {df['rep_id'].nunique()} reps")
    train(df, tag=tag, epochs=args.epochs)
