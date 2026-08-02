#!/usr/bin/env python3
"""End-to-end smoke test for the bahrain_rent_solution package.

The real notebook takes ~1.5-2 h (60+ model fits), so this script proves the
full pipeline works by executing the NOTEBOOK'S OWN CODE at reduced scale:

    5 folds -> 2 folds | 3 seeds -> 1 seed | 6 optuna trials -> 1
    3000 boosting rounds -> 150 | submission -> temporary file

Everything else is untouched: same feature engineering, same leak-safe fold
statistics, same four learners, same blending and premium calibration, same
submission logic. A typical run takes ~2-4 minutes.

It validates the checkpoints that matter (row/feature counts, OOF and test
prediction matrices, blend weights, calibration factor, submission file) and
exits 0 on success, 1 on failure.

    python smoke_test.py
"""
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)   # notebook paths (data.csv, ...) are relative to the package

NOTEBOOK = "final_solution_simplified.ipynb"

# scale-down replacements applied to the notebook's code (exact source strings)
SMOKE_SUBMISSION = os.path.join(tempfile.gettempdir(),
                                "bahrain_rent_smoke_submission.csv")
REPLACEMENTS = [
    ("SEEDS = (42, 1337, 7)", "SEEDS = (42,)"),
    ("N_SPLITS = 5", "N_SPLITS = 2"),
    ("KFold(n_splits=3,", "KFold(n_splits=2,"),
    ("n_trials=6", "n_trials=1"),
    ("MAX_ROUNDS = 3000", "MAX_ROUNDS = 150"),
    ('"submission_simplified.csv"', json.dumps(SMOKE_SUBMISSION)),
]


def build_smoke_source():
    """Notebook code cells -> one python source with scale-down replacements."""
    with open(NOTEBOOK) as f:
        nb = json.load(f)
    src = "\n\n# %%\n\n".join("".join(cell["source"]) for cell in nb["cells"]
                               if cell["cell_type"] == "code")
    for old, new in REPLACEMENTS:
        if old not in src:
            raise RuntimeError(f"expected notebook source not found: {old!r}")
        src = src.replace(old, new)
    return src


def main():
    print(f"bahrain_rent_solution — smoke test (reduced-scale full pipeline)")
    print(f"notebook: {NOTEBOOK} | temp submission: {SMOKE_SUBMISSION}\n")

    os.environ.setdefault("MPLBACKEND", "Agg")   # headless: no plot windows
    had_catboost_logs = os.path.isdir("catboost_info")

    src = build_smoke_source()
    namespace = {"__name__": "__main__"}

    start = time.time()
    try:
        exec(compile(src, NOTEBOOK, "exec"), namespace)
    except Exception as exc:
        print(f"\n[FAIL] pipeline raised {type(exc).__name__}: {exc}")
        return 1
    elapsed = time.time() - start

    import numpy as np
    import pandas as pd

    problems = []

    def expect(condition, message):
        print(f"[{'PASS' if condition else 'FAIL'}] {message}")
        if not condition:
            problems.append(message)

    # --- checkpoint 1: data cleaning ---------------------------------------
    model_df = namespace.get("model_df")
    expect(model_df is not None and len(model_df) == 10558,
           f"model rows = {None if model_df is None else len(model_df)} "
           f"(expected 10558 = 10,575 - 17 error rows)")

    # --- checkpoint 2: feature list -----------------------------------------
    all_features = namespace.get("ALL_FEATURES", [])
    cat_features = namespace.get("CATEGORICAL_FEATURES", [])
    expect(len(all_features) == 203 and len(cat_features) == 8,
           f"feature list = {len(all_features)} features, "
           f"{len(cat_features)} categorical (expected 203 / 8)")

    # --- checkpoint 3: out-of-fold + test predictions ------------------------
    oof = namespace.get("oof_predictions")
    testp = namespace.get("test_predictions")
    expect(oof is not None and oof.shape == (10558, 4) and not oof.isna().any().any(),
           f"OOF predictions {None if oof is None else oof.shape} complete for 4 learners")
    expect(testp is not None and testp.shape == (3527, 4),
           f"test predictions {None if testp is None else testp.shape} (expected (3527, 4))")

    # --- checkpoint 4: blend weights + calibration ---------------------------
    weights = namespace.get("blend_weights")
    expect(weights is not None and abs(float(np.sum(weights)) - 1.0) < 1e-6
           and (np.asarray(weights) >= -1e-9).all(),
           f"blend weights convex (sum=1, >=0): "
           f"{None if weights is None else np.round(weights, 3).tolist()}")
    factor = namespace.get("calibration_factor")
    expect(factor is not None and 0.9 < factor < 1.3,
           f"premium calibration factor = {factor}")

    # --- checkpoint 5: submission file ---------------------------------------
    if os.path.isfile(SMOKE_SUBMISSION):
        sub = pd.read_csv(SMOKE_SUBMISSION)
        sample = pd.read_csv("sample_submission.csv")
        expect(len(sub) == len(sample) == 3527
               and sub["rent"].notna().all()
               and (sub["rent"] >= 0).all()
               and sub["Property_id"].equals(sample["Property_id"]),
               f"submission: {len(sub)} rows, no NaN, rent >= 0, "
               f"ids match sample_submission")
    else:
        expect(False, f"submission file was not written to {SMOKE_SUBMISSION}")

    # --- cleanup --------------------------------------------------------------
    if os.path.isfile(SMOKE_SUBMISSION):
        os.remove(SMOKE_SUBMISSION)
    if not had_catboost_logs and os.path.isdir("catboost_info"):
        shutil.rmtree("catboost_info")   # CatBoost training logs from this run

    print()
    if problems:
        print(f"RESULT: {len(problems)} check(s) FAILED ({elapsed:.0f}s).")
        return 1
    print(f"RESULT: smoke test passed in {elapsed:.0f}s — the notebook's pipeline "
          f"runs end to end in this environment.")
    print("For the real result, run the full notebook (see run_all.sh output).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
