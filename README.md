# Whiff Line

A hobby model that projects MLB starting-pitcher strikeout totals, compares them to
sportsbook lines, and surfaces positive-EV edges. Runs itself on GitHub Actions,
hosted for free on GitHub Pages. See `methodology.html` for the model details.

## What's in here

```
index.html            Today's board (live picks)
performance.html       Full historical record
methodology.html       How the model works
assets/                Shared CSS/JS for the three pages
data/                  JSON files the pages read — Actions writes to these
scripts/                The actual pipeline (Python)
models/                 Trained model files land here (empty until first retrain)
.github/workflows/      The three scheduled jobs
```

## One-time setup (no local coding required)

1. **Create the repo.** On GitHub, click **New repository**, name it whatever
   you like (e.g. `whiff-line`), keep it **public** (Pages' free tier needs
   this, and it's also what makes the record honest — anyone can check it).
2. **Upload these files.** Easiest way with no command line: on the new
   repo's page, click **Add file → Upload files**, drag the whole folder in,
   commit.
3. **Turn on Pages.** Repo → **Settings → Pages** → under "Build and
   deployment," set Source to **Deploy from a branch**, branch **main**,
   folder **/ (root)**. Save. Your site will be live at
   `https://yourusername.github.io/whiff-line/` within a minute or two.
4. **Turn on Actions permissions.** Repo → **Settings → Actions → General**
   → under "Workflow permissions," select **Read and write permissions**.
   This lets the scheduled jobs commit updated data back to the repo.
5. **Run the first training pass manually.** Repo → **Actions** tab → click
   **Retrain** in the left sidebar → **Run workflow**. This builds
   `data/training_data.csv` and the model files from scratch. **Expect this
   to take a while the first time** — see the note below.
6. **Enter today's lines.** Edit `data/lines_today.json` right in the GitHub
   web UI (click the file → pencil icon) with today's date and whatever
   strikeout prop lines you're looking at. This step is manual by design —
   free odds APIs don't carry player props.
7. **Run Predict once manually** (Actions tab → **Predict** → Run workflow)
   to confirm it works, then let the schedules in `.github/workflows/*.yml`
   take over.

From here it runs itself: **Predict** refreshes the board several times a
day, **Reconcile** grades yesterday's picks every morning, **Retrain** rebuilds
the model weekly.

## Things you'll likely want to tune

- **The Retrain job pulls Statcast data for the full window since 2022 every
  single time**, which is slow (real talk: possibly multiple hours for a
  multi-year pull). Once you have a working `data/training_data.csv`, a
  quick win is editing `build_training_data.py` to only pull *new* dates
  since the last run and append, rather than rebuilding from scratch weekly.
- **Opponent lineup K% is approximated** with team-level season rates rather
  than the true starting nine — see the comment at the top of
  `build_training_data.py` for why, and where to plug in something better
  if you find a good free source of historical lineup cards.
- **Name matching** between MLB Stats API and Statcast (`playerid_lookup` in
  `predict_today.py`) is done by first/last name text match, which will
  occasionally whiff on suffixes (Jr., III) or accented characters. Worth
  hardening with MLBAM ID caching once you see it happen.
- **Lineup confidence tiers** (confirmed / recent-usage / team-average,
  described in `methodology.html`) are stubbed to a flat 1.0 in
  `predict_today.py` — wiring up the actual lineup-card lookup and the 1.0 /
  0.6 / 0.3 fallback logic is the natural next step.

## Running it locally instead of waiting on Actions

```
pip install -r requirements.txt
python scripts/build_training_data.py   # slow — pulls Statcast history
python scripts/train_model.py
python scripts/predict_today.py         # needs data/lines_today.json filled in
python scripts/reconcile.py             # grades anything from a finished game
```

All data sources (MLB Stats API, Baseball Savant via `pybaseball`,
Open-Meteo) are free and don't need an API key. Total cost to run this: $0.

## Disclaimer

Personal research project, not betting advice. Please gamble responsibly —
1-800-GAMBLER.
