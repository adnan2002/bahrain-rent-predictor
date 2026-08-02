#!/usr/bin/env python3
"""Fast compatibility check for the bahrain_rent_solution package.

Runs in seconds and verifies everything the notebook needs, WITHOUT running
the notebook itself:

  1. Python interpreter version
  2. Required distributions installed (compared against requirements.txt pins)
  3. Runtime imports of every module the pipeline actually uses
  4. All expected files present
  5. Input data integrity (shapes, key columns)
  6. Configuration file integrity (final_config.json keys)
  7. Geocoding mode resolution (.env -> OFFLINE flag, as the notebook does)
  8. Cached external data integrity (coordinates, POIs, coastline, anchors)
  9. Offline feature-build probe (the whole feature stack on a small sample,
     caches only — no network)

Exit code 0 = all hard checks passed (warnings allowed), 1 = at least one
failure. Run it with the interpreter you intend to run the notebook with:

    python check_setup.py
"""
import importlib
import importlib.metadata
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")   # same policy as the notebook itself

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)   # all paths in this package are relative to its own directory

FAILURES = 0
WARNINGS = 0


def passed(name, detail=""):
    print(f"[PASS] {name}" + (f" — {detail}" if detail else ""))


def warned(name, detail):
    global WARNINGS
    WARNINGS += 1
    print(f"[WARN] {name} — {detail}")


def failed(name, detail):
    global FAILURES
    FAILURES += 1
    print(f"[FAIL] {name} — {detail}")


def run(name, fn):
    """Run a hard check; anything raised becomes a FAIL line."""
    try:
        passed(name, fn() or "")
    except Exception as exc:
        failed(name, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 1. Python version
# ---------------------------------------------------------------------------
def check_python():
    v = sys.version_info
    if v < (3, 10):
        raise RuntimeError(f"Python {v.major}.{v.minor} is too old (need >= 3.10)")
    return f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})"


# ---------------------------------------------------------------------------
# 2. Distributions vs requirements.txt
# ---------------------------------------------------------------------------
def check_requirements():
    installed, missing, mismatched = [], [], []
    with open("requirements.txt") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    for line in lines:
        if "==" in line:
            dist, pin = line.split("==", 1)
        else:
            dist, pin = line, None
        try:
            found = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            missing.append(dist)
            continue
        if pin and found != pin:
            mismatched.append(f"{dist} {found} (pinned {pin})")
        else:
            installed.append(f"{dist} {found}")
    if missing:
        raise RuntimeError("not installed: " + ", ".join(missing)
                           + "  ->  pip install -r requirements.txt")
    if mismatched:
        warned("requirements pins", "; ".join(mismatched))
    return f"{len(installed) + len(mismatched)} distributions present"


# ---------------------------------------------------------------------------
# 3. Runtime imports (what the pipeline actually imports)
# ---------------------------------------------------------------------------
RUNTIME_IMPORTS = [
    "numpy", "pandas", "matplotlib.pyplot", "scipy.optimize",
    "sklearn", "lightgbm", "xgboost", "catboost", "optuna",
    "holidays", "hijridate", "shapely", "pyproj", "requests", "dotenv",
]


def check_imports():
    for module in RUNTIME_IMPORTS:
        importlib.import_module(module)
    return f"{len(RUNTIME_IMPORTS)} modules import cleanly"


# ---------------------------------------------------------------------------
# 4. Expected files
# ---------------------------------------------------------------------------
EXPECTED_FILES = [
    # notebook
    "final_solution_simplified.ipynb",
    # input data
    "data.csv", "test.csv", "sample_submission.csv",
    # configuration
    "final_config.json",
    # project code
    "feature_pipeline.py", "distance_features.py", "location_upgrades.py",
    "extended_pois.py",
    # cached external data
    "area_coordinates.csv", "tower_coordinates.csv", "governorate_stats.csv",
    "bahrain_coastline.geojson", "bahrain_pois.csv", "bahrain_pois_extended.csv",
    "strategic_anchors.csv",
    # packages / environment
    "requirements.txt", ".env",
]


def check_files():
    missing = [f for f in EXPECTED_FILES if not os.path.isfile(f)]
    if missing:
        raise RuntimeError("missing: " + ", ".join(missing))
    return f"all {len(EXPECTED_FILES)} expected files present"


# ---------------------------------------------------------------------------
# 5. Input data integrity
# ---------------------------------------------------------------------------
def check_input_data():
    import pandas as pd
    train = pd.read_csv("data.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")
    assert train.shape == (10578, 16), f"data.csv shape {train.shape}"
    assert test.shape == (3527, 15), f"test.csv shape {test.shape}"
    assert "rent" in train.columns and "rent" not in test.columns
    assert list(sample.columns) == ["Property_id", "rent"], sample.columns.tolist()
    assert len(sample) == len(test), "sample submission must cover every test row"
    assert sample["Property_id"].is_unique
    return f"train {train.shape} | test {test.shape} | sample {sample.shape}"


# ---------------------------------------------------------------------------
# 6. Configuration integrity
# ---------------------------------------------------------------------------
def check_config():
    with open("final_config.json") as f:
        cfg = json.load(f)
    for key in ["clean", "null", "te", "m_smooth", "num_feats", "cat_feats"]:
        assert key in cfg, f"final_config.json missing key '{key}'"
    assert cfg["clean"]["rent_cap"] == 10000
    return (f"rent_cap={cfg['clean']['rent_cap']}, m_smooth={cfg['m_smooth']}, "
            f"{len(cfg['num_feats'])} base numeric + {len(cfg['cat_feats'])} categorical")


# ---------------------------------------------------------------------------
# 7. Geocoding mode resolution (same rule the notebook uses)
# ---------------------------------------------------------------------------
def check_geocoding_mode():
    from dotenv import load_dotenv
    load_dotenv(".env")
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if key:
        return ("GOOGLE_MAPS_API_KEY loaded -> notebook runs with OFFLINE=False "
                "(online geocoding of unseen areas/towers enabled)")
    return ("no GOOGLE_MAPS_API_KEY in .env -> notebook runs with OFFLINE=True "
            "(cached data only — all caches verified below)")


# ---------------------------------------------------------------------------
# 8. Cached external data integrity
# ---------------------------------------------------------------------------
def check_caches():
    import pandas as pd
    area = pd.read_csv("area_coordinates.csv")
    tower = pd.read_csv("tower_coordinates.csv")
    gov = pd.read_csv("governorate_stats.csv")
    pois = pd.read_csv("bahrain_pois.csv")
    ext = pd.read_csv("bahrain_pois_extended.csv")
    anchors = pd.read_csv("strategic_anchors.csv")
    with open("bahrain_coastline.geojson") as f:
        coast = json.load(f)

    assert {"Area", "latitude", "longitude"} <= set(area.columns)
    assert len(area) >= 100, f"only {len(area)} geocoded areas"
    assert len(tower) == 24, f"expected 24 tower entries, got {len(tower)}"
    assert len(gov) == 4, f"expected 4 governorates, got {len(gov)}"
    assert {"mall", "international_school"} <= set(pois["category"])
    assert {"supermarket", "food", "school_all", "healthcare", "fitness",
            "beach"} <= set(ext["category"])
    assert len(anchors) == 5, f"expected 5 strategic anchors, got {len(anchors)}"
    assert coast.get("type") == "FeatureCollection" and len(coast["features"]) > 0
    return (f"{len(area)} areas, {len(tower)} towers, {len(gov)} governorates, "
            f"{len(pois)} POIs, {len(ext)} extended POIs, {len(anchors)} anchors, "
            f"{len(coast['features'])} coastline ways")


# ---------------------------------------------------------------------------
# 9. Offline feature-build probe (whole feature stack, caches only)
# ---------------------------------------------------------------------------
def check_feature_build():
    import pandas as pd
    from feature_pipeline import build_row_features

    train_sample = pd.read_csv("data.csv", nrows=30).dropna(subset=["Title"])
    test_sample = pd.read_csv("test.csv", nrows=10)

    fe_train = build_row_features(train_sample, offline=True)
    fe_test = build_row_features(test_sample, offline=True)

    # 211 columns on train (210 + rent), 210 on test — same as the full run
    assert fe_train.shape[1] == 211, f"train features: {fe_train.shape}"
    assert fe_test.shape[1] == 210, f"test features: {fe_test.shape}"
    for col in ["size_sqm", "latitude", "dist_to_coastline_km", "tower",
                "n_supermarket_within_2km", "dist_to_navy_base_km", "kw_view"]:
        assert col in fe_train.columns, f"missing feature column '{col}'"
    # geo features must actually be populated for the sample rows
    assert fe_train["latitude"].notna().all(), "area coordinates did not map"
    return (f"row features built offline: train {fe_train.shape}, "
            f"test {fe_test.shape}")


# ---------------------------------------------------------------------------
def main():
    print(f"bahrain_rent_solution — compatibility check ({HERE})\n")
    run("1. Python version", check_python)
    run("2. requirements.txt distributions", check_requirements)
    run("3. runtime imports", check_imports)
    run("4. expected files", check_files)
    run("5. input data", check_input_data)
    run("6. final_config.json", check_config)
    run("7. geocoding mode resolution", check_geocoding_mode)
    run("8. cached external data", check_caches)
    run("9. offline feature-build probe", check_feature_build)

    print()
    if FAILURES:
        print(f"RESULT: {FAILURES} check(s) FAILED, {WARNINGS} warning(s).")
        return 1
    print(f"RESULT: all checks passed ({WARNINGS} warning(s)). "
          f"Next: run smoke_test.py for an end-to-end verification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
