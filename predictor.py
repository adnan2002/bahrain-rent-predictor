"""Inference stack for the Bahrain rent predictor (joblib-exportable).

All logic here is pickled BY REFERENCE into the joblib artifact, so this
module (plus `feature_pipeline` and its helpers) must be importable wherever
the artifact is loaded.
"""
import json
import os

import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from feature_pipeline import (build_row_features, apply_stats,
                              group_features, KEYWORD_VOCAB)

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "final_config.json")) as _f:
    _CONFIG = json.load(_f)

calendar_numeric, calendar_categorical = group_features("G1_calendar")
extra_numeric, _ = group_features("G6_new")

interaction_features = ["te_area_x_size", "te_areapt_x_size",
                        "te_area_x_beds", "navy_x_dist"]
keyword_features = [f"kw_{word}" for word in KEYWORD_VOCAB]
market_features = ["te_area_ppsqm", "te_ppsqm_x_size", "te_areapt_ppsqm",
                   "te_subarea", "coast_x_seaview", "rot_xy_1", "rot_xy_2"]

NUMERIC_FEATURES = (_CONFIG["num_feats"] + calendar_numeric + extra_numeric
                    + interaction_features + keyword_features + market_features)
CATEGORICAL_FEATURES = _CONFIG["cat_feats"] + calendar_categorical

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LEARNER_NAMES = ["lgb_log", "xgb_log", "cb_log", "lgbq"]


def add_interaction_features(df):
    out = df.copy()
    out["te_area_x_size"] = out["te_area"] * out["log_size_sqm"]
    out["te_areapt_x_size"] = out["te_area_ptype"] * out["log_size_sqm"]
    out["te_area_x_beds"] = out["te_area"] * out["beds_num"]
    out["navy_x_dist"] = out["navy_approved"] * out["dist_to_navy_base_km"]
    return out


def prepare_model_matrix(df):
    X = add_interaction_features(df)[ALL_FEATURES].copy()
    for col in X.columns:
        if str(X[col].dtype) == "boolean":
            X[col] = X[col].astype("int8")
        elif str(X[col].dtype) == "Int64":
            X[col] = X[col].astype("float64")
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("missing").astype("category")
        if "missing" not in X[col].cat.categories:
            X[col] = X[col].cat.add_categories("missing")
    return X


def align_categories_with_train(X, X_train):
    X = X.copy()
    for col in CATEGORICAL_FEATURES:
        train_cats = X_train[col].cat.categories
        unseen_as_missing = X[col].where(X[col].isin(train_cats), "missing")
        X[col] = pd.Categorical(unseen_as_missing, categories=train_cats)
    return X


def predict_learner(name, model, X):
    """Learner prediction in BHD (log learners use expm1, quantile clips at 0)."""
    if name in ("lgb_log", "xgb_log", "cb_log"):
        return np.expm1(model.predict(X))
    if name == "lgbq":
        return np.clip(model.predict(X), 0, None)
    raise ValueError(name)


class BahrainRentPredictor:
    """End-to-end rent predictor for propertyfinder.bh listings.

    predict(raw_df) takes the raw competition columns (Property_id, Title,
    Area, Governorate, Property_type, Beds, Baths, Size, Amenities,
    Availability_date, Agency, Agent_name, ...) and returns monthly rent in
    BHD. Wraps: stateless row features -> leak-safe fold statistics ->
    4 seed-bagged learners -> SLSQP blend -> premium calibration -> clip.
    """

    def __init__(self, stats, models, blend_weights, calibration_factor,
                 reference_categories):
        self.stats = stats
        self.models = models                      # {learner: [seed models]}
        self.blend_weights = blend_weights        # np.array over LEARNER_NAMES
        self.calibration_factor = calibration_factor
        self.reference_categories = reference_categories  # col -> categories

    def predict(self, raw_df):
        features = build_row_features(raw_df, offline=True)
        X = prepare_model_matrix(apply_stats(features, self.stats))
        X = self._align_categories(X)
        blended = np.zeros(len(X))
        for name in LEARNER_NAMES:
            pred = np.mean([predict_learner(name, m, X)
                            for m in self.models[name]], axis=0)
            blended += self.blend_weights[LEARNER_NAMES.index(name)] * pred
        blended = blended.copy()
        blended[blended > 900] *= self.calibration_factor
        return np.clip(blended, 0, None)

    def _align_categories(self, X):
        X = X.copy()
        for col, cats in self.reference_categories.items():
            unseen = X[col].where(X[col].isin(cats), "missing")
            X[col] = pd.Categorical(unseen, categories=cats)
        return X
