"""
predict_today.py — runs several times a day via the predict workflow.

1. Pulls today's schedule + probable pitchers from MLB Stats API.
2. Pulls each probable pitcher's last ~45 days of Statcast pitches to build
   the same EWMA features used in training (causal, no look-ahead needed
   here since we want "as of right now").
3. Loads the saved LightGBM models and produces p10/p25/p50/p75/p90 +
   Poisson point estimate for each starter.
4. Fits a PCHIP curve through the quantiles to get P(strikeouts > line) for
   whatever line is in data/lines_today.json (entered manually — see
   methodology.html "Honest limitations").
5. Computes de-vigged edge and writes data/projections.json, ranked by edge.

If lines_today.json has no entry for a pitcher, that pitcher is still
scored (proj_p50 etc.) but shown without an edge, so the board still fills
in even on days you haven't entered every line yet.
"""
from __future__ import annotations
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA_DIR,
    MODELS_DIR,
    get_schedule,
    devig_two_way,
    load_json,
    save_json,
    utc_now_iso,
)

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
LOOKBACK_DAYS = 45


def load_models():
    point = lgb.Booster(model_file=str(MODELS_DIR / "point_poisson.txt"))
    quants = {
        q: lgb.Booster(model_file=str(MODELS_DIR / f"quantile_p{int(q * 100):02d}.txt"))
        for q in QUANTILES
    }
    feature_cols = load_json(MODELS_DIR / "feature_columns.json", None)
    if feature_cols is None:
        raise SystemExit("No trained models found — run train_model.py first.")
    return point, quants, feature_cols


def probable_starters(today: dt.date) -> list[dict]:
    games = get_schedule(today)
    starters = []
    for g in games:
        teams = g.get("teams", {})
        for side in ("home", "away"):
            team_info = teams.get(side, {})
            pitcher = team_info.get("probablePitcher")
            if not pitcher:
                continue
            opp_side = "away" if side == "home" else "home"
            starters.append({
                "pitcher_id": pitcher.get("id"),
                "pitcher": pitcher.get("fullName"),
                "team": team_info.get("team", {}).get("name"),
                "opponent": teams.get(opp_side, {}).get("team", {}).get("name"),
                "home_away": side,
                "game_pk": g.get("gamePk"),
                "venue": g.get("venue", {}).get("name"),
                "game_date": today.isoformat(),
            })
    return starters


def recent_features_for_pitcher(pitcher_name: str, as_of: dt.date) -> dict | None:
    """
    Pull a pitcher's recent Statcast pitches and compute the same EWMA
    features used at training time, as of right now. Falls back to None
    (skip the pitcher) if there isn't enough recent data — e.g. a rookie's
    first ever start, or a name-matching miss against Statcast's player_name
    field, which is a known rough edge worth hardening later.
    """
    from pybaseball import playerid_lookup, statcast_pitcher

    last, first = pitcher_name.split(" ")[-1], " ".join(pitcher_name.split(" ")[:-1])
    lookup = playerid_lookup(last, first)
    if lookup.empty:
        return None
    mlbam_id = int(lookup.iloc[0]["key_mlbam"])

    start = (as_of - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    end = (as_of - dt.timedelta(days=1)).isoformat()
    pitches = statcast_pitcher(start, end, mlbam_id)
    if pitches is None or pitches.empty:
        return None

    called_strike = pitches["description"].eq("called_strike")
    whiff = pitches["description"].isin(["swinging_strike", "swinging_strike_blocked"])
    csw_pct = (called_strike | whiff).mean()
    swstr_pct = whiff.mean()
    batters_faced = pitches["at_bat_number"].nunique()
    strikeouts = (pitches["events"] == "strikeout").sum()
    starts = pitches["game_date"].nunique()
    k_pct = strikeouts / batters_faced if batters_faced else np.nan
    bf_per_start = batters_faced / starts if starts else np.nan

    return {
        "csw_pct_ewma": csw_pct,
        "swstr_pct_ewma": swstr_pct,
        "k_pct_ewma": k_pct,
        "batters_faced_ewma": bf_per_start,
        "prior_k_pct": k_pct,   # season-to-date proxy; see train_model.py docstring
        "prior_weight": min(1.0, batters_faced / (batters_faced + 70)) if batters_faced else 0.0,
        "mlbam_id": mlbam_id,
    }


def prob_over_line(quantile_values: list[float], line: float) -> float:
    """PCHIP through (value, cumulative probability) points -> P(X > line)."""
    xs = quantile_values
    ys = QUANTILES
    if xs != sorted(xs):
        # quantile crossing (rare, small-sample models) — sort as a safe fallback
        pairs = sorted(zip(xs, ys))
        xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    curve = PchipInterpolator(xs, ys, extrapolate=True)
    cdf_at_line = float(np.clip(curve(line), 0.0, 1.0))
    return 1 - cdf_at_line


def run(today: dt.date | None = None):
    today = today or dt.date.today()
    point_model, quant_models, feature_cols = load_models()
    lines = load_json(DATA_DIR / "lines_today.json", {"lines": []})
    lines_by_pitcher = {row["pitcher"]: row for row in lines.get("lines", [])}

    picks = []
    for starter in probable_starters(today):
        feats = recent_features_for_pitcher(starter["pitcher"], today)
        if feats is None:
            continue
        X = pd.DataFrame([{c: feats[c] for c in feature_cols}])

        proj_p50_point = float(point_model.predict(X)[0])
        quantile_values = [float(quant_models[q].predict(X)[0]) for q in QUANTILES]
        # enforce monotonicity in case the independently-trained quantile
        # models cross slightly at the extremes
        quantile_values = list(np.maximum.accumulate(quantile_values))

        line_row = lines_by_pitcher.get(starter["pitcher"])
        pick = {
            **{k: v for k, v in starter.items() if k != "game_date" or True},
            "proj_point": round(proj_p50_point, 2),
            "proj_p10": round(quantile_values[0], 2),
            "proj_p25": round(quantile_values[1], 2),
            "proj_p50": round(quantile_values[2], 2),
            "proj_p75": round(quantile_values[3], 2),
            "proj_p90": round(quantile_values[4], 2),
            "lineup_confidence": 1.0,  # TODO: wire up the confirmed/estimated/fallback tiers
        }

        if line_row:
            line = line_row["line"]
            model_p_over = prob_over_line(quantile_values, line)
            book_p_over, book_p_under = devig_two_way(line_row["over_price"], line_row["under_price"])
            edge_over = model_p_over - book_p_over
            edge_under = (1 - model_p_over) - book_p_under
            if edge_over >= edge_under:
                pick.update(line=line, prob_over=round(model_p_over, 3), edge=round(edge_over, 3), side="OVER")
            else:
                pick.update(line=line, prob_over=round(model_p_over, 3), edge=round(edge_under, 3), side="UNDER")
        else:
            pick.update(line=None, prob_over=None, edge=None, side=None)

        picks.append(pick)

    # only surface positive-EV sides on the board, per the model's own rule
    ranked = sorted(
        [p for p in picks if p.get("edge") is not None and p["edge"] > 0],
        key=lambda p: p["edge"],
        reverse=True,
    )

    save_json(DATA_DIR / "projections.json", {
        "generated_at": utc_now_iso(),
        "all_starters_scored": len(picks),
        "picks": ranked,
    })

    # Upsert into the permanent history log that reconcile.py grades against.
    # Keyed by (date, pitcher_id) — later runs the same day overwrite earlier
    # ones, so the version on file is always the latest projection made
    # before first pitch. reconcile.py is what "locks" it in once the game
    # is final.
    history_path = DATA_DIR / "predictions_history.json"
    history = load_json(history_path, {"entries": {}})
    for p in picks:
        key = f"{today.isoformat()}::{p['pitcher_id']}"
        history["entries"][key] = {**p, "date": today.isoformat(), "logged_at": utc_now_iso(), "reconciled": False}
    save_json(history_path, history)

    print(f"Scored {len(picks)} starters, {len(ranked)} with a positive edge. Wrote data/projections.json")


if __name__ == "__main__":
    run()
