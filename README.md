# Bahrain Rent Prediction

End-to-end machine-learning pipeline for the Bahrain rent-prediction competition:
predict monthly rent (BHD) from propertyfinder.bh listings.

**Metric:** Mean Absolute Error (MAE) · **Leaderboard score:** 86.5 (previous best: 87)

The full solution lives in **`final.ipynb`** — a readable,
heavily-commented notebook that walks through the entire pipeline:

1. **Clean the target** — drop exactly the 17 data-error rows (`rent > 10,000`);
   dropping more is metric gaming (proven with an honest full-distribution CV protocol)
2. **Feature engineering** — ~200 stateless features: parsed sizes/beds/baths,
   calendar (holidays, Ramadan, availability), geography (coordinates, distances,
   POI densities, towers), governorate statistics, title mining, top-100 keyword flags
3. **Leak-safe statistics** — smoothed target encodings + rent-per-sqm encodings,
   listing counts, KNN amenities imputation — all fit *inside* each CV fold
4. **Modeling** — four fold-bagged boosters (LightGBM, XGBoost, CatBoost, quantile
   LightGBM), 5 folds × 3 seeds = 15 fits each; XGBoost tuned with Optuna
5. **Blending** — SLSQP-optimized convex weights in BHD space + premium-segment
   calibration (×~1.02 above 900 BHD)
6. **Submission** — `final_submission.csv`

## Repository layout

| Path | Purpose |
|---|---|
| `final.ipynb` | the complete pipeline, documented end to end |
| `data.csv` / `test.csv` / `final_submission.csv` | competition data |
| `final_config.json` | the validated recipe (feature lists, rent cap, smoothing) |
| `feature_pipeline.py` | all row-local transforms + leak-safe fold statistics |
| `distance_features.py` | coastline / CBD / airport / causeway / school / mall distances |
| `location_upgrades.py` | tower identity, building-level coords, views, floor, zone flags |
| `extended_pois.py` | OSM POI density counts + strategic-anchor distances |
| `area_coordinates.csv`, `tower_coordinates.csv`, `governorate_stats.csv`, `bahrain_coastline.geojson`, `bahrain_pois*.csv`, `strategic_anchors.csv` | cached external data (no network needed) |
| `requirements.txt` | pinned python dependencies |
| `check_setup.py` / `smoke_test.py` / `run_all.sh` | verification scripts (see below) |

## Setup

Requires Python 3.10+:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Verify the environment (minutes, not hours)

A full notebook run takes ~1.5–2 h, so the repo ships with two verification
scripts instead:

```bash
./run_all.sh            # runs both checks (or: python check_setup.py && python smoke_test.py)
```

- **`check_setup.py`** (seconds) — Python/packages vs `requirements.txt`, all
  files present, input data + config + caches intact, and an offline
  feature-build probe on a small sample
- **`smoke_test.py`** (~2 min) — executes the notebook's own code end to end at
  reduced scale (2 folds, 1 seed, 150 boosting rounds) and validates every
  checkpoint: 10,558 model rows, 203 features, OOF/test prediction matrices,
  convex blend weights, calibration factor, and a valid 3,527-row submission

## Full run

```bash
jupyter nbconvert --to notebook --execute final.ipynb
```

or open the notebook in Jupyter and *Run All*. The output is
`final_submission.csv` (gitignored).

## Offline geocoding (no API key required)

All external data (geocoding, POIs, coastline, governorate stats) is **cached in
this repo**, so the pipeline runs fully offline. The notebook resolves the mode
automatically from `.env`:

- `.env` with `GOOGLE_MAPS_API_KEY=...` present → `OFFLINE = False`
  (unseen areas/towers may be geocoded online and appended to the caches)
- no `.env` / no key → `OFFLINE = True` (cached data only)

Since `.env` is gitignored, a fresh clone always runs in offline mode.
Caches are always consulted first — online mode only ever geocodes entries
that are missing from the caches.
