"""
Train a LightGBM regression model to predict fluid_diff_ml.
Split is by rep_id to avoid data leakage.

Usage:
  py train.py --features cole_cole --sample 200
  py train.py --features raw --sample 0        # all data, raw features
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

OUTPUT_DIR = Path("C:/Users/tal-m/MyProject/output")
OUTPUT_DIR.mkdir(exist_ok=True)

NON_FEATURE = {"fluid_diff_ml", "t_i", "t_j", "rep_id", "seed",
               "fluid_t", "fluid_t+b", "fluid_diff", "gauss"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE]


def split_by_rep(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
    rep_ids = df["rep_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_val_idx, test_idx = next(gss.split(df, groups=rep_ids))

    df_tv = df.iloc[train_val_idx]
    df_test = df.iloc[test_idx]

    val_rel = val_frac / (1 - test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed)
    tr_idx, val_idx = next(gss2.split(df_tv, groups=df_tv["rep_id"].values))

    return df_tv.iloc[tr_idx], df_tv.iloc[val_idx], df_test


def train(df: pd.DataFrame, tag: str = "model"):
    feature_cols = get_feature_cols(df)
    print(f"Features: {len(feature_cols)}  |  Rows: {len(df)}")

    train_df, val_df, test_df = split_by_rep(df)
    print(f"Split — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print(f"Unique reps — train: {train_df['rep_id'].nunique()}, "
          f"val: {val_df['rep_id'].nunique()}, test: {test_df['rep_id'].nunique()}")

    X_train, y_train = train_df[feature_cols], train_df["fluid_diff_ml"]
    X_val,   y_val   = val_df[feature_cols],   val_df["fluid_diff_ml"]
    X_test,  y_test  = test_df[feature_cols],  test_df["fluid_diff_ml"]

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    print(f"\nTest  RMSE: {rmse:.4f} mL  MAE: {mae:.4f} mL  R²: {r2:.4f}")

    # predicted vs actual
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, preds, alpha=0.3, s=10)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual fluid_diff_ml (mL)")
    ax.set_ylabel("Predicted fluid_diff_ml (mL)")
    ax.set_title(f"{tag}  R²={r2:.3f}  RMSE={rmse:.3f} mL")
    fig.tight_layout()
    out_plot = OUTPUT_DIR / f"{tag}_pred_vs_actual.png"
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_plot}")

    # feature importance (top 30)
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    fi.head(30).plot.barh(ax=ax2)
    ax2.invert_yaxis()
    ax2.set_title(f"{tag} — Top 30 feature importances")
    fig2.tight_layout()
    out_fi = OUTPUT_DIR / f"{tag}_feature_importance.png"
    fig2.savefig(out_fi, dpi=150)
    plt.close(fig2)

    joblib.dump(model, OUTPUT_DIR / f"{tag}.pkl")
    print(f"Model saved to {OUTPUT_DIR / f'{tag}.pkl'}")
    return model, {"rmse": rmse, "mae": mae, "r2": r2}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=["raw", "cole_cole", "cole_cole_enhanced"], default="cole_cole")
    parser.add_argument("--sample", type=int, default=200,
                        help="Number of reps to sample (0 = all data)")
    args = parser.parse_args()

    tag = f"lgbm_{args.features}_n{args.sample if args.sample else 'all'}"

    from features import load_sample_features, load_all_features
    from data_loader import load_sample, load_all

    if args.features == "raw":
        if args.sample > 0:
            print(f"Loading raw features for {args.sample} reps...")
            df = load_sample(args.sample)
        else:
            print("Loading raw features for all reps...")
            df = load_all()
    elif args.features == "cole_cole_enhanced":
        mode = "cole_cole"  # enhanced = cole_cole + raw ΔZ, handled inside features.py
        if args.sample > 0:
            print(f"Extracting enhanced Cole-Cole+dZ features for {args.sample} reps...")
            df = load_sample_features(args.sample, mode=mode)
        else:
            df = load_all_features(mode=mode)
    else:
        if args.sample > 0:
            print(f"Extracting Cole-Cole features for {args.sample} reps...")
            df = load_sample_features(args.sample)
        else:
            print("Extracting Cole-Cole features for all reps...")
            df = load_all_features()

    print(f"Loaded {len(df)} rows from {df['rep_id'].nunique()} reps")
    train(df, tag=tag)
