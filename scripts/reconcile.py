"""
reconcile.py — runs once overnight via the reconcile workflow.

Walks data/predictions_history.json for entries that haven't been graded
yet, checks whether that game is final, and if so pulls the final boxscore,
grades the pick (hit / miss / push), and appends it to data/performance.json
— which is the permanent, append-only record the Performance page reads.
Nothing already in performance.json is ever edited or removed; this script
only appends.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA_DIR,
    get_boxscore,
    get_pitcher_strikeouts_from_boxscore,
    load_json,
    save_json,
    utc_now_iso,
)


def grade(actual_k: int, line: float | None, side: str | None) -> tuple[str, str]:
    if line is None or side is None:
        return "pending", "No line entered"
    if actual_k == line:
        return "push", "Push"
    over_hit = actual_k > line
    hit = (side == "OVER" and over_hit) or (side == "UNDER" and not over_hit)
    label = f"{side} {'hit' if hit else 'missed'}"
    return ("hit" if hit else "miss"), label


def run():
    history_path = DATA_DIR / "predictions_history.json"
    history = load_json(history_path, {"entries": {}})
    performance = load_json(DATA_DIR / "performance.json", {"generated_at": None, "log": []})

    newly_graded = 0
    for key, entry in history["entries"].items():
        if entry.get("reconciled"):
            continue
        game_pk = entry.get("game_pk")
        if not game_pk:
            continue

        try:
            box = get_boxscore(game_pk)
        except Exception as e:  # noqa: BLE001
            print(f"Skipping {key}, boxscore not ready: {e}")
            continue

        # Only grade once the game is actually final — a missing strikeout
        # count means it's still in progress, so leave it for next run.
        actual_k = get_pitcher_strikeouts_from_boxscore(box, entry["pitcher_id"])
        if actual_k is None:
            continue

        grade_code, grade_label = grade(actual_k, entry.get("line"), entry.get("side"))
        performance["log"].append({
            "date": entry["date"],
            "pitcher": entry["pitcher"],
            "opponent": entry["opponent"],
            "line": entry.get("line"),
            "proj_p50": entry.get("proj_p50"),
            "edge": entry.get("edge"),
            "actual_k": actual_k,
            "grade": grade_code,
            "grade_label": grade_label,
        })
        entry["reconciled"] = True
        newly_graded += 1

    performance["generated_at"] = utc_now_iso()
    save_json(DATA_DIR / "performance.json", performance)
    save_json(history_path, history)
    print(f"Reconciled {newly_graded} new result(s). Performance log now has {len(performance['log'])} entries.")


if __name__ == "__main__":
    run()
