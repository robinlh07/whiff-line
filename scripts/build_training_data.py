"""
build_training_data.py — assembles one row per starting-pitcher start since 2022.

Run this rarely (weekly, via the retrain workflow) — it pulls full-season
pitch-level Statcast data via `pybaseball`, which is slow (many minutes for
a multi-year expanding window). Output: data/training_data.csv, one row per
(pitcher, game_date) start with the target (actual strikeouts) and the raw
ingredients for the features described in methodology.html. EWMA smoothing
and the early-season prior are computed later, in train_model.py, so this
file can be reused for every retrain without re-pulling Statcast each time.

Known simplification (flagged here on purpose, see methodology.html
"Honest limitations"): opponent lineup K% vs handedness is approximated
with the *team's* season-to-date K% vs that handedness rather than a true
slot-weighted reconstruction of the actual lineup card, because historical
lineup cards aren't cheaply available for backtesting. Swap in real lineup
history here if you find a good free source.
"""
from __future__ import annotations
import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, get_schedule, get_boxscore, get_pitcher_strikeouts_from_boxscore  # noqa: E402

TRAINING_START = dt.date(2022, 3, 1)
OUTPUT_PATH = DATA_DIR / "training_data.csv"


def pull_statcast_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Pitch-level Statcast data for a date range. Requires `pybaseball`."""
    from pybaseball import statcast  # imported lazily so the rest of the
    # module can be unit-tested without the (heavy) pybaseball dependency

    df = statcast(start_dt=start.isoformat(), end_dt=end.isoformat())
    return df


def csw_and_swstr_by_start(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch-level data into one row per (pitcher, game_date)."""
    if pitch_df.empty:
        return pd.DataFrame(
            columns=["pitcher", "game_date", "pitches", "csw_pct", "swstr_pct", "strikeouts_statcast"]
        )

    called_strike = pitch_df["description"].eq("called_strike")
    whiff = pitch_df["description"].isin(["swinging_strike", "swinging_strike_blocked"])
    swing_desc = [
        "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
        "hit_into_play", "foul_bunt",
    ]
    is_swing = pitch_df["description"].isin(swing_desc)

    pitch_df = pitch_df.assign(is_csw=called_strike | whiff, is_whiff=whiff, is_swing=is_swing)

    grouped = pitch_df.groupby(["pitcher", "game_date"]).agg(
        pitches=("description", "size"),
        csw=("is_csw", "sum"),
        whiffs=("is_whiff", "sum"),
        swings=("is_swing", "sum"),
        batters_faced=("at_bat_number", "nunique"),
    ).reset_index()

    grouped["csw_pct"] = grouped["csw"] / grouped["pitches"]
    grouped["swstr_pct"] = grouped["whiffs"] / grouped["pitches"]
    return grouped[["pitcher", "game_date", "pitches", "batters_faced", "csw_pct", "swstr_pct"]]


def build(start: dt.date, end: dt.date, out_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    print(f"Pulling Statcast pitch data {start} → {end} (this is the slow part)...")
    pitches = pull_statcast_range(start, end)
    per_start = csw_and_swstr_by_start(pitches)
    print(f"Built {len(per_start)} pitcher-starts from Statcast.")

    # Actual strikeouts (target) come straight from Statcast events, which is
    # simpler than re-fetching every boxscore for a multi-year backfill.
    if not pitches.empty:
        so_events = pitches[pitches["events"] == "strikeout"]
        so_counts = so_events.groupby(["pitcher", "game_date"]).size().rename("strikeouts").reset_index()
        per_start = per_start.merge(so_counts, on=["pitcher", "game_date"], how="left")
        per_start["strikeouts"] = per_start["strikeouts"].fillna(0).astype(int)

        # basic context columns available straight from Statcast
        ctx_cols = pitches.groupby(["pitcher", "game_date"]).agg(
            player_name=("player_name", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            inning_topbot=("inning_topbot", "first"),
        ).reset_index()
        per_start = per_start.merge(ctx_cols, on=["pitcher", "game_date"], how="left")
        per_start["is_home"] = per_start["inning_topbot"].eq("Bot")
        per_start["opponent"] = per_start.apply(
            lambda r: r["away_team"] if r["is_home"] else r["home_team"], axis=1
        )

    per_start = per_start.sort_values(["pitcher", "game_date"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_start.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(per_start)} rows).")
    return per_start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=TRAINING_START.isoformat())
    parser.add_argument("--end", default=dt.date.today().isoformat())
    args = parser.parse_args()
    build(dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end))
