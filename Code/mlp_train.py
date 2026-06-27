"""
MLP regression model to predict fluid_diff_ml from raw impedance spectra.

Input: Z1_real/imag, Z2_real/imag, ΔZ_real/imag at 30 frequencies
       for all 4 electrode pairs → 4 × 6 × 30 = 720 features + dt_steps = 721

Architecture: MLP with batch norm and dropout.
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

from features import load_sample_features, load_all_features

OUTPUT_DIR = Path("C:/Users/tal-m/MyProject/output")
OUTPUT_DIR.mkdir(exist_ok=True)

NON_FEATURE = {"fluid_diff_ml", "t_i", "t_j", "rep_id", "seed",
               "fluid_t", "fluid_t+b", "fluid_diff", "gauss"}


def get_feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE]


def split_by_rep(df, val_frac=0.15, test_frac=0.15, seed=42):
    rep_ids = df["rep_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tv_idx, te_idx = next(gss.split(df, groups=rep_ids))
    df_tv, df_te = df.iloc[tv_idx], df.iloc[te_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac / (1 - test_frac), random_state=seed)
    tr_idx, va_idx = next(gss2.split(df_tv, groups=df_tv["rep_id"].values))
    return df_tv.iloc[tr_idx], df_tv.iloc[va_idx], df_te


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int] = [512, 256, 128], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train(df: pd.DataFrame, tag: str = "mlp", epochs: int = 150, batch_size: int = 256, lr: float = 1e-3):
    feature_cols = get_feature_cols(df)
    print(f"Features: {len(feature_cols)}  |  Rows: {len(df)}")

    train_df, val_df, test_df = split_by_rep(df)
    print(f"Split — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values.astype(np.float32))
    X_val   = scaler.transform(val_df[feature_cols].values.astype(np.float32))
    X_test  = scaler.transform(test_df[feature_cols].values.astype(np.float32))

    y_train = train_df["fluid_diff_ml"].values.astype(np.float32)
    y_val   = val_df["fluid_diff_ml"].values.astype(np.float32)
    y_test  = test_df["fluid_diff_ml"].values.astype(np.float32)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MLP(input_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_count = 0
    patience_limit = 20

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_ds)
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
        preds = model(torch.tensor(X_test).to(device)).cpu().numpy()

    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    print(f"\nTest  RMSE: {rmse:.4f} mL  MAE: {mae:.4f} mL  R²: {r2:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, preds, alpha=0.3, s=10)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual fluid_diff_ml (mL)")
    ax.set_ylabel("Predicted fluid_diff_ml (mL)")
    ax.set_title(f"{tag}  R²={r2:.3f}  RMSE={rmse:.3f} mL")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_pred_vs_actual.png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {OUTPUT_DIR / f'{tag}_pred_vs_actual.png'}")

    torch.save(best_state, OUTPUT_DIR / f"{tag}.pt")
    joblib.dump(scaler, OUTPUT_DIR / f"{tag}_scaler.pkl")
    print(f"Model saved to {OUTPUT_DIR / f'{tag}.pt'}")
    return model, {"rmse": rmse, "mae": mae, "r2": r2}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    tag = f"mlp_raw_n{args.sample if args.sample else 'all'}"

    if args.sample > 0:
        print(f"Loading raw features for {args.sample} reps...")
        df = load_sample_features(args.sample, mode="raw")
    else:
        df = load_all_features(mode="raw")

    print(f"Loaded {len(df)} rows from {df['rep_id'].nunique()} reps")
    train(df, tag=tag, epochs=args.epochs)
