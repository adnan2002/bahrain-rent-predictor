"""
Leak-safe feature pipeline for the Bahrain rent competition (metric: MAE).

Two-layer design
----------------
1. ``build_row_features(df, ...)`` — **stateless, row-local** transforms.
   Every output value depends only on that row plus fixed external data
   (geocoding caches, POI lists, governorate statistics). Safe to run once
   on train and test independently.

2. ``fit_stats(train_df, y_log, ...)`` / ``apply_stats(df, stats)`` —
   everything that *learns from other rows*: listing counts, target-encoding
   maps (with smoothing), Amenities KNN imputation, availability median.
   Inside cross-validation these are fit on the train folds only, so no
   information leaks from validation rows into their own features.

``clean_target(df, strategy)`` implements the CV-validated outlier / duplicate
strategies (train set only — never drop test rows).

Reuses the existing project modules: ``distance_features``,
``location_upgrades``, ``extended_pois``.
"""
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
AREA_COORDS_FILE = os.path.join(_HERE, "area_coordinates.csv")
GOV_STATS_FILE = os.path.join(_HERE, "governorate_stats.csv")

# Approximate scrape date of the listings (availability dates cluster in
# Aug-Oct 2025). Only used as a fixed reference point for "days until
# available" — the model sees a monotone transform of the date either way.
REFERENCE_DATE = pd.Timestamp("2025-07-01")

TARGET = "rent"
ID_COL = "Property_id"

# Columns that must never enter the model matrix.
RAW_DROP_COLS = ["Property_id", "Offer", "URL", "Title", "Agent_name",
                 "Availability_date", "avail_dt", "Beds", "Baths", "Size",
                 "keywords"]

# The obsolete "Central Governorate" (abolished Sep 2014) remapped by Area.
CENTRAL_AREA_TO_GOVERNORATE = {
    "Tubli": "Capital", "Jid Ali": "Capital", "Sanad": "Capital",
    "Isa Town": "Capital", "Sitra": "Capital", "Jurdab": "Capital",
    "Maameer": "Capital", "Eker": "Capital", "A'Ali": "Northern",
    "Salmabad": "Northern", "Nuwaidrat": "Southern",
}

# Top-100 title tokens (frequency on train titles, stopwords removed) — validated
# in the feature lab: one-hot keywords beat TF-IDF SVD embeddings (−1.7 MAE).
KEYWORD_VOCAB = [
    'balcony', 'view', 'sea', 'pool', 'location', 'private', 'garden', 'prime',
    'bright', 'ewa', 'gym', '2bhk', 'kitchen', 'room', 'luxurious', 'facilities',
    'access', 'city', 'beach', 'near', 'great', 'studio', 'maid', 'amp',
    'compound', 'elegant', 'high', 'floor', 'bhk', 'amenities', 'family', 'wifi',
    'housekeeping', 'closed', 'navy', 'affordable', 'juffair', 'area', 'amazing',
    '1bhk', 'price', 'stunning', '3bhk', 'approved', 'friendly', 'deal',
    'living', 'internet', 'cozy', 'seef', 'close', '2br', 'views', 'stylish',
    'best', 'unlimited', 'maids', 'all', 'bedrooms', 'hot', 'pet', 'offer',
    '3br', 'renovated', 'bed', 'duplex', 'large', 'amwaj', 'balconies', 'apt',
    'pets', 'mall', 'building', 'huge', 'house', 'kids', 'hidd', 'open', 'saar',
    'free', '1br', 'gas', 'parking', 'swimming', 'keeping', '4br', 'well', 'one',
    'service', 'premium', 'terrace', 'full', 'specious', 'furniture',
    'apartments', 'allowed', 'home', 'quiet', 'two', '4bhk',
]
_KW_STOP = set(
    "for rent in and the with of a an is are on at to from by new bedroom"
    " bathrooms flat apartment villa bahrain fully semi furnished inclusive"
    " exclusive brand luxury spacious modern beautiful nice good big located"
    " rent rental monthly year available now call contact us property".split())


# ============================================================================
# Row-local feature builders (stateless)
# ============================================================================
def add_size_features(df):
    out = df.copy()
    sqm = (out["Size"].astype("string")
           .str.extract(r"/\s*([\d,]+)\s*sqm")[0]
           .str.replace(",", "", regex=False).astype(float))
    out["size_sqm"] = sqm
    out["log_size_sqm"] = np.log1p(sqm)
    return out


def add_beds_baths_features(df):
    out = df.copy()
    beds = out["Beds"].astype("string")
    out["beds_is_studio"] = beds.str.contains("studio", case=False, na=False)
    out["beds_has_maid"] = beds.str.contains("Maid", case=False, na=False)
    out["beds_is_plus"] = beds.str.contains(r"\+", na=False)
    out["beds_num"] = beds.str.extract(r"(\d+)")[0].astype(float)
    out.loc[out["beds_is_studio"], "beds_num"] = 0

    baths = out["Baths"].astype("string")
    out["baths_is_plus"] = baths.str.contains(r"\+", na=False)
    # "none" has no digits -> NaN (missing, not a real 0-bathroom listing)
    out["baths_num"] = baths.str.extract(r"(\d+)")[0].astype(float)

    beds_safe = out["beds_num"].replace(0, np.nan)
    baths_safe = out["baths_num"].replace(0, np.nan)
    out["sqm_per_bedroom"] = out["size_sqm"] / beds_safe
    out["sqm_per_bathroom"] = out["size_sqm"] / baths_safe
    out["beds_per_sqm"] = out["beds_num"] / out["size_sqm"]
    out["baths_per_sqm"] = out["baths_num"] / out["size_sqm"]
    # capped bedroom count, used as a categorical key for target encoding
    out["beds_cat"] = out["beds_num"].clip(0, 7).astype("Int64")
    return out


def add_calendar_features(df, reference_date=REFERENCE_DATE):
    """Season, month, quarter, Ramadan, holiday proximity + availability extras."""
    import holidays
    from hijridate import Gregorian

    out = df.copy()
    avail = pd.to_datetime(out["Availability_date"], format="%d %b %Y",
                           errors="coerce")
    out["avail_dt"] = avail
    out["season"] = avail.dt.month.map(
        lambda m: pd.NA if pd.isna(m) else ("Summer" if 4 <= m <= 10 else "Winter"))

    unique_dates = avail.dropna().unique()
    if len(unique_dates):
        years = range(unique_dates.year.min() - 1, unique_dates.year.max() + 2)
        holiday_dates = np.array(
            sorted(holidays.country_holidays("BH", years=years).keys()),
            dtype="datetime64[D]")
        dtf = pd.DataFrame(index=unique_dates)
        dtf["month"] = dtf.index.month
        dtf["quarter"] = dtf.index.quarter
        dtf["is_ramadan"] = [int(Gregorian(d.year, d.month, d.day).to_hijri().month == 9)
                             for d in dtf.index]
        dtf["days_to_nearest_holiday"] = [
            int(np.abs(holiday_dates - np.datetime64(d.date())).astype(int).min())
            for d in dtf.index]
        for col in dtf.columns:
            out[col] = avail.map(dtf[col]).astype("Int64")
    else:
        for col in ["month", "quarter", "is_ramadan", "days_to_nearest_holiday"]:
            out[col] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["avail_year"] = avail.dt.year.astype("Int64")

    # availability extras (signal documented in null_handling.md)
    out["days_to_availability"] = (avail - reference_date).dt.days.astype("float64")
    out["is_available_now"] = (avail <= reference_date).astype("float64")
    out.loc[avail.isna(), "is_available_now"] = np.nan
    out["availability_missing"] = avail.isna().astype("int8")
    return out


def add_area_coordinates(df, offline=False, api_key=None):
    """Area -> (latitude, longitude), cached; geocodes unseen areas on miss."""
    import requests

    out = df.copy()
    coords = (pd.read_csv(AREA_COORDS_FILE)
              if os.path.exists(AREA_COORDS_FILE)
              else pd.DataFrame(columns=["Area", "latitude", "longitude"]))
    missing = sorted(set(out["Area"].dropna().unique()) - set(coords["Area"]))
    if missing and not offline:
        if api_key is None:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(_HERE, ".env"))
            api_key = os.environ["GOOGLE_MAPS_API_KEY"]
        rows = []
        for area in missing:
            r = requests.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"address": f"{area}, Bahrain", "key": api_key},
                timeout=15).json()
            loc = (r["results"][0]["geometry"]["location"]
                   if r.get("status") == "OK" else {})
            rows.append({"Area": area, "latitude": loc.get("lat"),
                         "longitude": loc.get("lng")})
        coords = pd.concat([coords, pd.DataFrame(rows)], ignore_index=True)
        coords.to_csv(AREA_COORDS_FILE, index=False)
    coords = coords.set_index("Area")
    out["latitude"] = out["Area"].map(coords["latitude"])
    out["longitude"] = out["Area"].map(coords["longitude"])
    return out


def _fetch_gov_stats():
    """Governorate population density + non-Bahraini ratio from data.gov.bh."""
    import requests
    ODS_API = "https://www.data.gov.bh/api/explore/v2.1/catalog/datasets"

    def latest_snapshot(dataset_id, select):
        params = {"select": f"year, {select}", "order_by": "year DESC", "limit": 100}
        records = requests.get(f"{ODS_API}/{dataset_id}/records",
                               params=params, timeout=30).json()["results"]
        latest_year = max(r["year"] for r in records)
        return [r for r in records if r["year"] == latest_year]

    density = {r["governorate"]: r["value"]
               for r in latest_snapshot("04-population-density-by-governorate",
                                        "governorate, value")
               if r["governorate"] != "Total"}
    pop = pd.DataFrame(latest_snapshot(
        "02-population-by-governorate-nationality-sex",
        "governorate, nationality, population"))
    pop_by_nat = pop.pivot_table(index="governorate", columns="nationality",
                                 values="population", aggfunc="sum")
    nbr = (pop_by_nat["Non-Bahraini"] / pop_by_nat.sum(axis=1)).to_dict()
    stats = pd.DataFrame({"governorate_current": list(density),
                          "pop_density_km2": list(density.values())})
    stats["non_bahraini_ratio"] = stats["governorate_current"].map(nbr)
    stats.to_csv(GOV_STATS_FILE, index=False)
    return stats


def add_governorate_stats(df, offline=False):
    out = df.copy()
    if os.path.exists(GOV_STATS_FILE):
        stats = pd.read_csv(GOV_STATS_FILE)
    elif offline:
        raise FileNotFoundError(
            f"{GOV_STATS_FILE} missing and offline=True — run once online.")
    else:
        stats = _fetch_gov_stats()
    stats = stats.set_index("governorate_current")

    out["governorate_current"] = out["Governorate"].str.replace(
        " Governorate", "", regex=False)
    central = out["governorate_current"] == "Central"
    out.loc[central, "governorate_current"] = out.loc[central, "Area"].map(
        CENTRAL_AREA_TO_GOVERNORATE)
    out["pop_density_km2"] = out["governorate_current"].map(stats["pop_density_km2"])
    out["non_bahraini_ratio"] = out["governorate_current"].map(stats["non_bahraini_ratio"])
    return out


FLAG_PATTERNS = {
    "is_private_pool":    r"private pool",
    "is_private_garden":  r"(?:private )?garden",
    "is_penthouse":       r"penthouse",
    "has_maid_room":      r"\bmaid",
    "is_sea_view":        r"sea ?view|seaview|seafront",
    "has_beach_access":   r"beach",
    "is_pet_friendly":    r"\bpet",
    "is_brand_new":       r"brand new",
    "has_housekeeping":   r"housekeep|houskeep",
    "has_balcony":        r"\bbalcon",
    "has_gym":            r"\bgym|fitness",
    "has_wifi":           r"\bwifi\b|internet|fiber",
    "is_compound":        r"compound",
    "has_parking":        r"parking",
    "has_terrace":        r"terrace",
    "is_high_floor":      r"high floor",
    "is_renovated":       r"renovat|upgraded|refurbish",
    "is_duplex":          r"duplex",
    "has_walk_in_closet": r"walk.?in|wardrobe|closet",
    "navy_approved":      r"navy",
    "is_first_resident":  r"first resident",
}
_TONE_WORDS = ["luxur", "spacious", "modern", "elegant", "prime", "stylish",
               "deluxe", "exclusive"]


def add_keyword_flags(df, vocab=KEYWORD_VOCAB):
    """One-hot flags for the top title keywords (feature-lab winner: −1.7 MAE)."""
    out = df.copy()
    tok = (out["Title"].fillna("").str.lower()
           .str.replace(r"[^a-z0-9 ]", " ", regex=True))
    token_lists = tok.str.split().apply(
        lambda xs: frozenset(x for x in xs if x not in _KW_STOP and len(x) > 2))
    for w in vocab:
        out[f"kw_{w}"] = token_lists.apply(lambda s, w=w: int(w in s)).astype("int8")
    return out


def add_subarea(df):
    """Split 'Sub Area, Main Area' -> subarea / area_main (36% of rows have one)."""
    out = df.copy()
    parts = out["Area"].fillna("").str.split(",")
    out["subarea"] = (parts.str[:-1].str.join(",").str.strip()
                      .replace("", "none"))
    out["area_main"] = parts.str[-1].str.strip()
    return out


def add_geo_extras(df):
    """Coastline x sea-view interaction + rotated coordinates (lab-validated)."""
    out = df.copy()
    out["coast_x_seaview"] = out["is_sea_view"] * out["dist_to_coastline_km"]
    out["rot_xy_1"] = out["latitude"] + out["longitude"]
    out["rot_xy_2"] = out["latitude"] - out["longitude"]
    return out


def add_title_features(df):
    out = df.copy()
    raw = out["Title"].fillna("").astype(str)
    t = raw.str.lower()
    for name, pat in FLAG_PATTERNS.items():
        out[name] = t.str.contains(pat, regex=True).astype("int8")
    out["furnishing_level"] = np.select(
        [t.str.contains(r"semi[ -]?furnished"), t.str.contains(r"\bfurnished\b")],
        [1, 2], default=0).astype("int8")
    out["view_type"] = np.select(
        [t.str.contains(r"sea ?view|seaview|seafront"), t.str.contains(r"city view")],
        ["sea_view", "city_view"], default="none")
    out["luxury_tone_score"] = np.sum([t.str.contains(w) for w in _TONE_WORDS],
                                      axis=0).astype("int8")
    # title statistics
    out["title_length"] = raw.str.len().astype("float64")
    out["title_words"] = raw.str.split().str.len().astype("float64")
    letters = raw.str.replace(r"[^A-Za-z]", "", regex=True)
    caps = raw.str.replace(r"[^A-Z]", "", regex=True).str.len()
    out["title_caps_ratio"] = (caps / letters.str.len().replace(0, np.nan)).fillna(0.0)
    out["title_exclamations"] = raw.str.count("!").astype("float64")
    out["Amenities_missing"] = out["Amenities"].isna().astype("int8")
    return out


def build_row_features(df, *, offline=False, with_extended_pois=True,
                       api_key=None):
    """All stateless, row-local features. Safe to run on train/test separately.

    Requires the raw competition columns (Title, Area, Governorate, Beds,
    Baths, Size, Availability_date, Amenities; URL optional but recommended
    for tower extraction).
    """
    from distance_features import add_distance_features
    from location_upgrades import add_location_upgrades

    out = df.copy()
    out = add_size_features(out)
    out = add_beds_baths_features(out)
    out = add_calendar_features(out)
    out = add_area_coordinates(out, offline=offline, api_key=api_key)
    out = add_distance_features(out, lat_col="latitude", lng_col="longitude")
    out = add_governorate_stats(out, offline=offline)
    out = add_title_features(out)
    # tower identity, building-level coords + distances, view_type upgrade,
    # floor number, POI density, freehold / artificial-island flags
    out = add_location_upgrades(out, geocode=not offline, api_key=api_key)
    out["has_floor_num"] = out["floor_num"].notna().astype("int8")
    if with_extended_pois:
        from extended_pois import add_extended_poi_features, add_anchor_features
        out = add_extended_poi_features(out, lat_col="lat_fine",
                                        lng_col="lng_fine", offline=offline)
        out = add_anchor_features(out, lat_col="lat_fine", lng_col="lng_fine",
                                  offline=offline, api_key=api_key)
    out = add_keyword_flags(out)
    out = add_subarea(out)
    out = add_geo_extras(out)
    return out


# ============================================================================
# Target cleaning strategies (TRAIN ONLY — never drop test rows)
# ============================================================================
def clean_target(df, *, rent_cap=10_000, rent_floor=None, rps_cap=None,
                 rps_floor=None, winsorize=False, dedupe_url=False):
    """Apply an outlier/duplicate strategy. Returns a filtered copy.

    rent_cap     : drop rows with rent > cap (or clip to cap if winsorize=True)
    rent_floor   : drop rows with rent < floor
    rps_cap      : drop rows with rent/size_sqm > cap
    rps_floor    : drop rows with rent/size_sqm < floor
    winsorize    : clip rent at rent_cap instead of dropping those rows
    dedupe_url   : drop later occurrences of duplicate URLs
    """
    out = df.copy()
    if dedupe_url and "URL" in out.columns:
        out = out.drop_duplicates(subset=["URL"], keep="first")
    rps = out[TARGET] / out["size_sqm"]
    keep = pd.Series(True, index=out.index)
    if rent_cap is not None and not winsorize:
        keep &= out[TARGET] <= rent_cap
    if rent_floor is not None:
        keep &= out[TARGET] >= rent_floor
    if rps_cap is not None:
        keep &= rps <= rps_cap
    if rps_floor is not None:
        keep &= rps >= rps_floor
    out = out.loc[keep]
    if winsorize and rent_cap is not None:
        out = out.copy()
        out[TARGET] = out[TARGET].clip(upper=rent_cap)
    return out


# ============================================================================
# Learned statistics (fit on train folds only)
# ============================================================================
TE_SPECS = {
    "te_area":       ["Area"],
    "te_area_ptype": ["Area", "Property_type"],
    "te_agency":     ["Agency"],
    "te_tower":      ["tower"],
    "te_area_beds":  ["Area", "beds_cat"],
    "te_subarea":    ["subarea"],
}


def _ppsqm_map(df_, keys, m=30):
    """Smoothed log(rent/sqm) encoding — the sharpest local-market signal (lab)."""
    rps = df_["rent"] / df_["size_sqm"]
    y = np.log1p(rps)
    gm = float(y.mean())
    kd = df_[keys].astype(str).agg("||".join, axis=1)
    g = pd.DataFrame({"k": kd, "y": y}).groupby("k")["y"].agg(["mean", "count"])
    return gm, (g["count"] * g["mean"] + m * gm) / (g["count"] + m)


def _te_key(X, cols):
    kd = X[cols].copy()
    for c in cols:
        kd[c] = kd[c].astype(str).fillna("missing")
    return kd.agg("||".join, axis=1)


def _smoothed_map(keys, y, m):
    gm = y.mean()
    s = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["mean", "count"])
    return gm, (s["count"] * s["mean"] + m * gm) / (s["count"] + m)


@dataclass
class FoldStats:
    te_maps: dict = field(default_factory=dict)   # name -> (map Series, global mean)
    agent_counts: pd.Series = None
    agency_counts: pd.Series = None
    area_counts: pd.Series = None
    amenities_mode: str = "nan"                   # "nan" | "median" | "knn"
    amenities_median: float = np.nan
    amenities_knn: object = None
    amenities_knn_cols: list = None
    avail_median: float = np.nan
    avail_impute: bool = False
    ppsqm_area: tuple = None      # (global_mean, map Series) for log(rent/sqm) per Area
    ppsqm_areapt: tuple = None    # same per Area x Property_type


def _amenities_design(df_):
    """Design matrix for the Amenities KNN imputer (recipe from null_handling.md)."""
    from sklearn.preprocessing import StandardScaler
    num = df_[["beds_num", "baths_num", "size_sqm"]].astype(float).copy()
    num[:] = StandardScaler().fit_transform(num.fillna(num.median()))
    cats = pd.get_dummies(
        df_[["governorate_current", "Property_type", "Include_w_e", "Area", "Agency"]]
        .astype(str).fillna("missing"))
    return pd.concat([num.reset_index(drop=True),
                      cats.reset_index(drop=True)], axis=1)


def fit_stats(train_df, y_log, *, te_names=("te_area", "te_area_ptype", "te_agency"),
              m_smooth=30, amenities_mode="nan", avail_impute=False):
    """Learn all row-dependent statistics from a training (fold) dataframe."""
    from sklearn.impute import KNNImputer

    st = FoldStats(amenities_mode=amenities_mode, avail_impute=avail_impute)
    for name in te_names:
        cols = TE_SPECS[name]
        gm, enc = _smoothed_map(_te_key(train_df, cols), y_log, m_smooth)
        st.te_maps[name] = (enc, gm)
    st.agent_counts = train_df["Agent_name"].value_counts() if "Agent_name" in train_df else None
    st.agency_counts = train_df["Agency"].value_counts()
    st.area_counts = train_df["Area"].value_counts()
    st.avail_median = float(train_df["days_to_availability"].median())
    st.amenities_median = float(train_df["Amenities"].median())
    st.ppsqm_area = _ppsqm_map(train_df, ["Area"], m_smooth)
    st.ppsqm_areapt = _ppsqm_map(train_df, ["Area", "Property_type"], m_smooth)
    if amenities_mode == "knn":
        design = _amenities_design(train_df)
        imp = KNNImputer(n_neighbors=5, weights="distance")
        imp.fit(pd.concat([design, train_df["Amenities"].reset_index(drop=True)],
                          axis=1))
        st.amenities_knn = imp
        st.amenities_knn_cols = design.columns.tolist()
    return st


def apply_stats(df, st):
    """Attach learned-statistic features (counts, TE, imputations)."""
    out = df.copy()
    if st.agent_counts is not None:
        out["Agent_listing_count"] = (out["Agent_name"].map(st.agent_counts)
                                      .fillna(0).astype(int))
    out["Agency_listing_count"] = out["Agency"].map(st.agency_counts).fillna(0).astype(int)
    out["area_listing_count"] = out["Area"].map(st.area_counts).fillna(0).astype(int)
    if st.avail_impute:
        out["days_to_availability"] = out["days_to_availability"].fillna(st.avail_median)
        out["is_available_now"] = out["is_available_now"].fillna(
            float(st.avail_median <= 0))
    if st.amenities_mode == "median":
        out["Amenities"] = out["Amenities"].fillna(st.amenities_median)
    elif st.amenities_mode == "knn":
        design = _amenities_design(out)
        design = design.reindex(columns=st.amenities_knn_cols, fill_value=0)
        block = pd.concat([design, out["Amenities"].reset_index(drop=True)], axis=1)
        out["Amenities"] = st.amenities_knn.transform(block)[:, -1]
    for name, (enc, gm) in st.te_maps.items():
        out[name] = _te_key(out, TE_SPECS[name]).map(enc).fillna(gm)
    if st.ppsqm_area is not None:
        gm_a, enc_a = st.ppsqm_area
        out["te_area_ppsqm"] = out["Area"].astype(str).map(enc_a).fillna(gm_a)
        out["te_ppsqm_x_size"] = out["te_area_ppsqm"] * out["log_size_sqm"]
        gm_p, enc_p = st.ppsqm_areapt
        key = out["Area"].astype(str) + "||" + out["Property_type"].astype(str)
        out["te_areapt_ppsqm"] = key.map(enc_p).fillna(gm_p)
    return out


# ============================================================================
# Feature groups (used by the notebook's CV ladder)
# ============================================================================
CAT_BASE = ["Property_type", "Include_w_e", "Area", "Governorate", "Agency"]

FEATURE_GROUPS = {
    "G0_base": dict(
        num=["size_sqm", "beds_num", "baths_num", "sqm_per_bedroom",
             "sqm_per_bathroom", "beds_per_sqm", "baths_per_sqm",
             "Amenities", "Amenities_missing", "beds_is_studio",
             "beds_has_maid", "beds_is_plus", "baths_is_plus",
             "Agent_listing_count", "Agency_listing_count",
             "area_listing_count"],
        cat=CAT_BASE),
    "G1_calendar": dict(
        num=["month", "quarter", "is_ramadan", "days_to_nearest_holiday",
             "avail_year"],
        cat=["season"]),
    "G2_geo": dict(
        num=["latitude", "longitude", "dist_to_coastline_km",
             "dist_to_manama_cbd_km", "dist_to_airport_km",
             "dist_to_causeway_km", "dist_to_nearest_international_school_km",
             "dist_to_nearest_mall_km"],
        cat=[]),
    "G3_gov": dict(
        num=["pop_density_km2", "non_bahraini_ratio"],
        cat=["governorate_current"]),
    "G4_title": dict(
        num=list(FLAG_PATTERNS) + ["furnishing_level", "luxury_tone_score"],
        cat=["view_type"]),
    "G5_tower_poi": dict(
        num=["has_tower", "lat_fine", "lng_fine", "floor_num", "has_floor_num",
             "n_schools_within_2km", "n_malls_within_3km",
             "is_freehold_zone", "is_artificial_island",
             "dist_to_coastline_bldg_km", "dist_to_manama_cbd_bldg_km",
             "dist_to_airport_bldg_km", "dist_to_causeway_bldg_km",
             "dist_to_nearest_international_school_bldg_km",
             "dist_to_nearest_mall_bldg_km"],
        cat=["tower"]),
    "G6_new": dict(
        num=["days_to_availability", "is_available_now", "availability_missing",
             "title_length", "title_words", "title_caps_ratio",
             "title_exclamations", "log_size_sqm"],
        cat=[]),
    "G7_external": dict(
        num=["n_supermarket_within_2km", "n_food_within_2km",
             "n_school_all_within_2km", "n_healthcare_within_2km",
             "n_fitness_within_2km", "dist_to_nearest_beach_km",
             "dist_to_navy_base_km", "dist_to_bapco_km", "dist_to_alba_km",
             "dist_to_university_km", "dist_to_bfh_km"],
        cat=[]),
    "G8_te": dict(
        num=["te_area", "te_area_ptype", "te_agency"],
        cat=[]),
    # --- feature-lab winners (rounds 1-5, fast-LGB judged, −2.9 total) ---------
    "G9_keywords": dict(
        num=[f"kw_{w}" for w in KEYWORD_VOCAB],
        cat=[]),
    "G10_market": dict(
        num=["te_area_ppsqm", "te_ppsqm_x_size", "te_areapt_ppsqm",
             "te_subarea", "coast_x_seaview", "rot_xy_1", "rot_xy_2"],
        cat=[]),
}


def group_features(*groups):
    """(numeric_cols, cat_cols) for the union of the given group names."""
    num, cat = [], []
    for g in groups:
        spec = FEATURE_GROUPS[g]
        num += [c for c in spec["num"] if c not in num]
        cat += [c for c in spec["cat"] if c not in cat]
    return num, cat


def add_tfidf_svd(train_df, test_df=None, *, n_components=8, text_col="Title",
                  max_features=6000, random_state=42):
    """Dense title embeddings: TF-IDF (1-2 grams, sublinear) + TruncatedSVD.

    Added at the modeling stage (validated in the XGB feature screen: −0.8 MAE;
    first 8 components are the sweet spot — 16/32 overfit). Unsupervised, fitted
    on train titles only. Returns (train_aug, test_aug, svd_cols).
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                          min_df=3, sublinear_tf=True, lowercase=True)
    T_tr = vec.fit_transform(train_df[text_col].fillna(""))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    cols = [f"tfidf_svd_{i}" for i in range(n_components)]
    tr = train_df.join(pd.DataFrame(svd.fit_transform(T_tr),
                                    columns=cols, index=train_df.index))
    te = None
    if test_df is not None:
        T_te = vec.transform(test_df[text_col].fillna(""))
        te = test_df.join(pd.DataFrame(svd.transform(T_te),
                                       columns=cols, index=test_df.index))
    return tr, te, cols
