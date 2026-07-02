"""
train_model.py — feature engineering + model training.

Reads data/training_data.csv (built by build_training_data.py), engineers
the EWMA form features and the Marcel-style early-season prior described in
methodology.html, then trains:
  - one LightGBM regressor with a Poisson objective (the point estimate)
  - five LightGBM quantile regressors at p10/p25/p50/p75/p90

Models are saved to models/*.txt (LightGBM's native format — no pickling,
so no version-compat headaches loading them back in predict_today.py).

Run weekly via the retrain workflow, on the full expanding window.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, MODELS_DIR, save_json  # noqa: E402

TRAINING_PATH = DATA_DIR / "training_data.csv"
FEATURE_COLS = [
    "csw_pct_ewma",
    "swstr_pct_ewma",
    "k_pct_ewma",
    "batters_faced_ewma",
    "prior_k_pct",
    "prior_weight",
]
TARGET_COL = "strikeouts"
HALF_LIFE_STARTS = 5


def ewma_by_pitcher(df: pd.DataFrame, col: str, half_life: int = HALF_LIFE_STARTS) -> pd.Series:
    """
    Exponentially-weighted, causal (no look-ahead) rolling average per pitcher.
    Shifted by one start so a given row's feature only uses starts *before* it.
    """
    return (
        df.groupby("pitcher")[col]
        .apply(lambda s: s.shift(1).ewm(halflife=half_life, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["pitcher", "game_date"]).copy()
    df["k_pct_raw"] = df["strikeouts"] / df["batters_faced"].replace(0, np.nan)

    df["csw_pct_ewma"] = ewma_by_pitcher(df, "csw_pct")
    df["swstr_pct_ewma"] = ewma_by_pitcher(df, "swstr_pct")
    df["k_pct_ewma"] = ewma_by_pitcher(df, "k_pct_raw")
    df["batters_faced_ewma"] = ewma_by_pitcher(df, "batters_faced")

    # Early-season prior: this season's cumulative K% so far (a stand-in for
    # a true Marcel projection — swap in last year's real Marcel/park-adjusted
    # rate here if you have one), blended in by sample size.
    df["season"] = pd.to_datetime(df["game_date"]).dt.year
    df["cum_bf_season"] = df.groupby(["pitcher", "season"])["batters_faced"].cumsum().shift(1)
    df["cum_k_season"] = df.groupby(["pitcher", "season"])["strikeouts"].cumsum().shift(1)
    df["prior_k_pct"] = (df["cum_k_season"] / df["cum_bf_season"]).fillna(df["k_pct_ewma"])
    df["prior_weight"] = (df["cum_bf_season"].fillna(0) / (df["cum_bf_season"].fillna(0) + 70)).fillna(0)

    # drop rows with no usable history yet (first start of a pitcher's career in-window)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return df


def train(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    point_model = lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    point_model.fit(X, y)
    point_model.booster_.save_model(str(MODELS_DIR / "point_poisson.txt"))

    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    for q in quantiles:
        qm = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        qm.fit(X, y)
        qm.booster_.save_model(str(MODELS_DIR / f"quantile_p{int(q * 100):02d}.txt"))

    save_json(MODELS_DIR / "feature_columns.json", FEATURE_COLS)
    save_json(
        MODELS_DIR / "training_meta.json",
        {
            "trained_on_rows": len(df),
            "date_range": [str(df["game_date"].min()), str(df["game_date"].max())],
            "half_life_starts": HALF_LIFE_STARTS,
        },
    )
    print(f"Trained on {len(df)} starts. Models saved to {MODELS_DIR}")


if __name__ == "__main__":
    if not TRAINING_PATH.exists():
        raise SystemExit(
            f"{TRAINING_PATH} not found — run build_training_data.py first."
        )
    raw = pd.read_csv(TRAINING_PATH, parse_dates=["game_date"])
    featured = add_features(raw)
    train(featured)
