"""
Tier-1 location & text feature upgrades for Bahrain property data.

Adds (given a df with Title, URL, Area, latitude, longitude):
  tower                  - canonical building/project name (categorical, "none")
  has_tower              - 1 if a known tower/project was identified
  lat_fine / lng_fine    - tower coords when known, else area-centroid coords
  dist_*_bldg (x6)       - distance features recomputed at building resolution
  view_type (overwritten)- sea/lagoon/golf/pool/city/none (was 3 levels)
  floor_num              - extracted "12th floor" -> 12.0 (NaN if absent)
  n_schools_within_2km   - POI density counts (from bahrain_pois.csv)
  n_malls_within_3km
  is_freehold_zone       - designated freehold area (expat ownership -> demand)
  is_artificial_island   - reclaimed mega-project island
"""
import os
import re

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from distance_features import add_distance_features

_HERE = os.path.dirname(os.path.abspath(__file__))
TOWER_COORDS_FILE = os.path.join(_HERE, "tower_coordinates.csv")
POIS_CSV = os.path.join(_HERE, "bahrain_pois.csv")

# ---------------------------------------------------------------- towers ----
# Canonical tower/project patterns (curated from Title + URL exploration).
# key = canonical name, value = regex (matched on lowercase title or url slug)
TOWER_PATTERNS = {
    "Bannai Tower":            r"bannai",
    "Juffair Heights":         r"juffair heights",
    "Fontana Suites":          r"fontana suites",
    "Fontana Gardens":         r"fontana ?garden",
    "Fontana Infinity":        r"fontana infinity",
    "Fontana Court":           r"fontana court",
    "Fontana Towers":          r"fon?tana tower",
    "Fontana (generic)":       r"\bfontana\b|\bfonatana\b",  # catch-all; keep after specific Fontana patterns
    "Onyx Residence":          r"onyx",
    "Catamaran Tower":         r"catamaran",
    "Reef Residence":          r"reef residence",
    "Harbour Heights":         r"harbo?ur heights",
    "Hala Plaza":              r"hala plaza",
    "Sukoon Tower":            r"sukoon",
    "Ventura Tower":           r"ventura",
    "Orchid Residence":        r"or[hc]id",
    "Ambassador Residence":    r"ambassador",
    "Elite Residence":         r"elite residence",
    "Star Residence":          r"star residence",
    "Silver Tower":            r"silver tower",
    "Era View Tower":          r"era view tower",
    "The Address Residences":  r"the address|address residence",
    "Marassi Shores":          r"marassi[- ]shores",
    "Marassi Park":            r"marassi[- ]park",
    "Marassi Residences":      r"marassi[- ]residences|marassi residence",
    "Marassi Bay":             r"marassi[- ]bay",
}
_TOWER_RE = {k: re.compile(v) for k, v in TOWER_PATTERNS.items()}
BBOX = (25.55, 50.35, 26.45, 50.90)  # Bahrain bounds for geocode validation


def extract_tower(title, url=""):
    """Canonical tower/project name from Title (then URL slug), else 'none'."""
    t = (title or "").lower()
    for name, rx in _TOWER_RE.items():
        if rx.search(t):
            return name
    slug = (url or "").lower()
    if slug:
        for name, rx in _TOWER_RE.items():
            if rx.search(slug):
                return name
    return "none"


def _geocode_towers(tower_areas, api_key=None):
    """tower_areas: {tower: modal_area}. Returns {tower: (lat, lng)} w/ cache."""
    cache = (pd.read_csv(TOWER_COORDS_FILE).set_index("tower")
             if os.path.exists(TOWER_COORDS_FILE)
             else pd.DataFrame(columns=["tower", "latitude", "longitude"])
             .set_index("tower"))
    missing = [t for t in tower_areas if t not in cache.index]
    if missing:
        if api_key is None:
            load_dotenv(os.path.join(_HERE, ".env"))
            api_key = os.environ["GOOGLE_MAPS_API_KEY"]
        rows = []
        for t in missing:
            r = requests.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"address": f"{t}, {tower_areas[t]}, Bahrain",
                        "key": api_key}, timeout=15).json()
            lat = lng = np.nan
            if r["status"] == "OK":
                loc = r["results"][0]["geometry"]["location"]
                if BBOX[0] <= loc["lat"] <= BBOX[2] and BBOX[1] <= loc["lng"] <= BBOX[3]:
                    lat, lng = loc["lat"], loc["lng"]
            rows.append({"tower": t, "latitude": lat, "longitude": lng})
        cache = pd.concat([cache, pd.DataFrame(rows).set_index("tower")])
        cache.reset_index().to_csv(TOWER_COORDS_FILE, index=False)
    return cache["latitude"].to_dict(), cache["longitude"].to_dict()


# ------------------------------------------------------------- text bits ----
def extract_view(title):
    t = (title or "").lower()
    if re.search(r"sea ?view|seaview|seafront|sea view", t):
        return "sea_view"
    if re.search(r"lagoon|canal|marina", t):
        return "lagoon_view"
    if re.search(r"golf", t):
        return "golf_view"
    if re.search(r"pool view|pool ?side", t):
        return "pool_view"
    if re.search(r"city view", t):
        return "city_view"
    return "none"


def extract_floor(title):
    t = (title or "").lower()
    m = (re.search(r"(\d+)(?:st|nd|rd|th)\s*floor", t)
         or re.search(r"floor\s*(\d+)", t))
    return float(m.group(1)) if m else np.nan


# ------------------------------------------------------------- zone flags ---
FREEHOLD_KEYWORDS = [
    "amwaj", "reef island", "bahrain bay", "juffair", "seef", "durrat",
    "riffa views", "diyar", "dilmunia", "marassi", "abraj al lulu",
]
ARTIFICIAL_ISLAND_KEYWORDS = [
    "amwaj", "diyar", "reef island", "bahrain bay", "dilmunia", "durrat",
    "marassi", "bahrain harbour",
]


def _kw_flag(area, keywords):
    a = (area or "").lower()
    return int(any(k in a for k in keywords))


# ------------------------------------------------------------------ main ----
def add_location_upgrades(df, geocode=True, api_key=None):
    """Return df copy with all upgrade columns added."""
    out = df.copy()

    # towers
    url = out["URL"] if "URL" in out.columns else pd.Series("", index=out.index)
    out["tower"] = [extract_tower(t, u) for t, u in zip(out["Title"], url)]
    out["has_tower"] = (out["tower"] != "none").astype("int8")

    # building-level coordinates (fallback: area centroid)
    out["lat_fine"] = out["latitude"]
    out["lng_fine"] = out["longitude"]
    towers = sorted(t for t in out["tower"].unique() if t != "none")
    if towers and geocode:
        modal_area = (out[out["tower"] != "none"].groupby("tower")["Area"]
                      .agg(lambda s: s.mode().iloc[0]).to_dict())
        lat_map, lng_map = _geocode_towers(modal_area, api_key)
        m = out["tower"] != "none"
        tlat = out.loc[m, "tower"].map(lat_map)
        tlng = out.loc[m, "tower"].map(lng_map)
        ok = (tlat.notna() & tlng.notna()).to_numpy()
        idx = tlat.index[ok]
        out.loc[idx, "lat_fine"] = tlat.to_numpy()[ok]
        out.loc[idx, "lng_fine"] = tlng.to_numpy()[ok]

    # building-resolution distance features
    tmp = add_distance_features(out, lat_col="lat_fine", lng_col="lng_fine")
    dist_cols = [c for c in tmp.columns if c.startswith("dist_")]
    out[[c.replace("_km", "_bldg_km") for c in dist_cols]] = tmp[dist_cols]

    # text upgrades
    out["view_type"] = [extract_view(t) for t in out["Title"]]
    out["floor_num"] = [extract_floor(t) for t in out["Title"]]

    # POI density counts (haversine, vectorized)
    pois = pd.read_csv(POIS_CSV)
    schools = pois.loc[pois["category"] == "international_school",
                       ["latitude", "longitude"]].to_numpy()
    malls = pois.loc[pois["category"] == "mall",
                     ["latitude", "longitude"]].to_numpy()
    out["n_schools_within_2km"] = _count_within(out, schools, 2.0)
    out["n_malls_within_3km"] = _count_within(out, malls, 3.0)

    # zone flags
    out["is_freehold_zone"] = [ _kw_flag(a, FREEHOLD_KEYWORDS) for a in out["Area"]]
    out["is_artificial_island"] = [_kw_flag(a, ARTIFICIAL_ISLAND_KEYWORDS)
                                   for a in out["Area"]]
    out["is_freehold_zone"] = out["is_freehold_zone"].astype("int8")
    out["is_artificial_island"] = out["is_artificial_island"].astype("int8")
    return out


def _count_within(df, pts, radius_km):
    """Count POIs within radius_km of each row's lat/lng (haversine)."""
    lat1 = np.radians(df["latitude"].to_numpy(float))[:, None]
    lng1 = np.radians(df["longitude"].to_numpy(float))[:, None]
    lat2 = np.radians(pts[:, 0])[None, :]
    lng2 = np.radians(pts[:, 1])[None, :]
    dlat, dlng = lat2 - lat1, lng2 - lng1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    dist = 2 * 6371 * np.arcsin(np.sqrt(h))
    return (dist <= radius_km).sum(axis=1)
