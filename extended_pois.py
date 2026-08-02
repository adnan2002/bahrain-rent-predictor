"""
Extended POI & strategic-anchor features for Bahrain property data.

Two feature families, computed for any (lat, lng) in Bahrain:

1. OSM Overpass POI density counts (free, no API key), cached in
   ``bahrain_pois_extended.csv``:
     - n_supermarket_within_2km   (supermarket / hypermarket / convenience)
     - n_food_within_2km          (restaurant / fast_food / cafe)
     - n_school_all_within_2km    (all schools, not just international)
     - n_healthcare_within_2km    (hospital / clinic / pharmacy)
     - n_fitness_within_2km       (gym / fitness_centre)
     - dist_to_nearest_beach_km   (natural=beach nodes/ways)

2. Strategic employment/anchor distances (geocoded once via Google Maps,
   cached in ``strategic_anchors.csv``; hardcoded fallbacks if offline):
     - dist_to_navy_base_km       (US Naval Support Activity, Juffair —
                                   pairs with the `navy_approved` title flag)
     - dist_to_bapco_km           (Bapco refinery, Sitra)
     - dist_to_alba_km            (Aluminium Bahrain, Askar)
     - dist_to_university_km      (University of Bahrain, Sakhir)
     - dist_to_bfh_km             (Bahrain Financial Harbour)

Usage:
    from extended_pois import add_extended_poi_features, add_anchor_features
    df = add_extended_poi_features(df, lat_col="lat_fine", lng_col="lng_fine")
    df = add_anchor_features(df, lat_col="lat_fine", lng_col="lng_fine")

Distances are WGS84 geodesic kilometres. All network responses are cached to
CSV on first use; later runs (and ``offline=True``) read only the caches.
"""
import os

import numpy as np
import pandas as pd
import requests
from pyproj import Geod

_HERE = os.path.dirname(os.path.abspath(__file__))
EXT_POIS_CSV = os.path.join(_HERE, "bahrain_pois_extended.csv")
ANCHORS_CSV = os.path.join(_HERE, "strategic_anchors.csv")

_GEOD = Geod(ellps="WGS84")

# Bahrain bounding box (south, west, north, east)
_BBOX = (25.40, 50.15, 26.70, 50.95)

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_AMENITY_RE = "^(supermarket|restaurant|fast_food|cafe|school|hospital|clinic|pharmacy|gym)$"
_OVERPASS_QUERY = (
    f"[out:json][timeout:180];("
    f'node["amenity"~"{_AMENITY_RE}"]{_BBOX};'
    f'way["amenity"~"{_AMENITY_RE}"]{_BBOX};'
    f'node["shop"~"^(supermarket|hypermarket|convenience)$"]{_BBOX};'
    f'way["shop"~"^(supermarket|hypermarket)$"]{_BBOX};'
    f'node["natural"="beach"]{_BBOX};'
    f'way["natural"="beach"]{_BBOX};'
    f'node["leisure"~"^(fitness_centre)$"]{_BBOX};'
    f'way["leisure"~"^(fitness_centre)$"]{_BBOX};'
    f");out center;"
)

# category -> predicate on (tags dict)
def _categorize(tags):
    a = tags.get("amenity", "")
    shop = tags.get("shop", "")
    nat = tags.get("natural", "")
    lei = tags.get("leisure", "")
    if nat == "beach":
        return "beach"
    if a == "supermarket" or shop in ("supermarket", "hypermarket", "convenience"):
        return "supermarket"
    if a in ("restaurant", "fast_food", "cafe"):
        return "food"
    if a == "school":
        return "school_all"
    if a in ("hospital", "clinic", "pharmacy"):
        return "healthcare"
    if a == "gym" or lei == "fitness_centre":
        return "fitness"
    return None


# name -> (geocode query, fallback (lat, lng))
ANCHORS = {
    "navy_base": ("US Naval Support Activity Bahrain, Juffair, Bahrain",
                  (26.2150, 50.6047)),
    "bapco": ("Bapco Refining, Sitra, Bahrain",
              (26.1426, 50.6246)),
    "alba": ("Aluminium Bahrain, Askar, Bahrain",
             (25.9967, 50.5442)),
    "university": ("University of Bahrain, Zallaq, Bahrain",
                   (26.0489, 50.5103)),
    "bfh": ("Bahrain Financial Harbour, Manama, Bahrain",
            (26.2385, 50.5694)),
}
_BH_BBOX = (25.55, 50.35, 26.45, 50.90)  # sanity bounds for geocode results


# ----------------------------------------------------------------- fetch ----
def _fetch_extended_pois(path=EXT_POIS_CSV):
    """Query Overpass for Bahrain POIs and cache as CSV."""
    r = requests.post(_OVERPASS_URL, data={"data": _OVERPASS_QUERY},
                      timeout=240, headers={"User-Agent": "extended-pois/1.0"})
    r.raise_for_status()
    rows = []
    for el in r.json().get("elements", []):
        cat = _categorize(el.get("tags", {}))
        if cat is None:
            continue
        if el["type"] == "node":
            lat, lng = el.get("lat"), el.get("lon")
        else:  # way -> centroid from "center"
            c = el.get("center") or {}
            lat, lng = c.get("lat"), c.get("lon")
        if lat is None or lng is None:
            continue
        rows.append({"category": cat, "latitude": lat, "longitude": lng})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def _load_extended_pois(offline=False, path=EXT_POIS_CSV):
    if os.path.exists(path):
        return pd.read_csv(path)
    if offline:
        raise FileNotFoundError(
            f"{path} missing and offline=True — run once without offline.")
    return _fetch_extended_pois(path)


def _load_anchors(offline=False, api_key=None, path=ANCHORS_CSV):
    """{anchor_name: (lat, lng)}, geocoded once and cached to CSV."""
    cache = (pd.read_csv(ANCHORS_CSV).set_index("anchor")
             if os.path.exists(ANCHORS_CSV)
             else pd.DataFrame(columns=["anchor", "latitude", "longitude"]).set_index("anchor"))
    missing = [a for a in ANCHORS if a not in cache.index]
    if missing and not offline:
        if api_key is None:
            try:
                from dotenv import load_dotenv
                load_dotenv(os.path.join(_HERE, ".env"))
                api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
            except Exception:
                api_key = None
        rows = []
        for a in missing:
            query, fallback = ANCHORS[a]
            lat, lng = fallback
            if api_key:
                try:
                    r = requests.get(
                        "https://maps.googleapis.com/maps/api/geocode/json",
                        params={"address": query, "key": api_key},
                        timeout=15).json()
                    if r.get("status") == "OK":
                        loc = r["results"][0]["geometry"]["location"]
                        if (_BH_BBOX[0] <= loc["lat"] <= _BH_BBOX[2]
                                and _BH_BBOX[1] <= loc["lng"] <= _BH_BBOX[3]):
                            lat, lng = loc["lat"], loc["lng"]
                except Exception:
                    pass  # keep fallback
            rows.append({"anchor": a, "latitude": lat, "longitude": lng})
        cache = pd.concat([cache, pd.DataFrame(rows).set_index("anchor")])
        cache.reset_index().to_csv(ANCHORS_CSV, index=False)
    elif missing:  # offline: use hardcoded fallbacks
        rows = [{"anchor": a, "latitude": ANCHORS[a][1][0], "longitude": ANCHORS[a][1][1]}
                for a in missing]
        cache = pd.concat([cache, pd.DataFrame(rows).set_index("anchor")])
    return {a: (cache.loc[a, "latitude"], cache.loc[a, "longitude"]) for a in ANCHORS}


# ------------------------------------------------------------- distances ----
def _geodesic_km(lat, lng, target):
    tlat, tlng = target
    _, _, m = _GEOD.inv(np.full_like(lng, tlng), np.full_like(lat, tlat),
                        np.asarray(lng, float), np.asarray(lat, float))
    return m / 1000.0


def _count_within(lat, lng, pts, radius_km, chunk=1024):
    """Vectorized haversine count of pts within radius_km of each (lat, lng)."""
    if len(pts) == 0:
        return np.zeros(len(lat), dtype=int)
    pts = np.asarray(pts, dtype=float)
    lat2, lng2 = np.radians(pts[:, 0]), np.radians(pts[:, 1])
    out = np.zeros(len(lat), dtype=int)
    for s in range(0, len(lat), chunk):
        la1 = np.radians(lat[s:s + chunk])[:, None]
        lo1 = np.radians(lng[s:s + chunk])[:, None]
        dlat, dlng = lat2[None, :] - la1, lng2[None, :] - lo1
        h = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(lat2)[None, :] * np.sin(dlng / 2) ** 2
        out[s:s + chunk] = (2 * 6371 * np.arcsin(np.sqrt(h)) <= radius_km).sum(axis=1)
    return out


def _min_dist_km(lat, lng, pts, chunk=1024):
    if len(pts) == 0:
        return np.full(len(lat), np.nan)
    pts = np.asarray(pts, dtype=float)
    lat2, lng2 = np.radians(pts[:, 0]), np.radians(pts[:, 1])
    out = np.full(len(lat), np.inf)
    for s in range(0, len(lat), chunk):
        la1 = np.radians(lat[s:s + chunk])[:, None]
        lo1 = np.radians(lng[s:s + chunk])[:, None]
        dlat, dlng = lat2[None, :] - la1, lng2[None, :] - lo1
        h = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(lat2)[None, :] * np.sin(dlng / 2) ** 2
        out[s:s + chunk] = (2 * 6371 * np.arcsin(np.sqrt(h))).min(axis=1)
    return out


# ------------------------------------------------------------- features -----
COUNT_RADIUS_KM = {
    "supermarket": 2.0,
    "food": 2.0,
    "school_all": 2.0,
    "healthcare": 2.0,
    "fitness": 2.0,
}


def add_extended_poi_features(df, lat_col="lat_fine", lng_col="lng_fine",
                              offline=False):
    """Add OSM POI density counts + nearest-beach distance."""
    out = df.copy()
    pois = _load_extended_pois(offline=offline)
    valid = out[lat_col].notna() & out[lng_col].notna()
    lat = out.loc[valid, lat_col].to_numpy(float)
    lng = out.loc[valid, lng_col].to_numpy(float)

    for cat, radius in COUNT_RADIUS_KM.items():
        pts = pois.loc[pois["category"] == cat, ["latitude", "longitude"]].to_numpy()
        col = f"n_{cat}_within_{int(radius)}km"
        out[col] = 0
        out.loc[valid, col] = _count_within(lat, lng, pts, radius)

    beach_pts = pois.loc[pois["category"] == "beach", ["latitude", "longitude"]].to_numpy()
    out["dist_to_nearest_beach_km"] = np.nan
    if len(beach_pts):
        out.loc[valid, "dist_to_nearest_beach_km"] = np.round(
            _min_dist_km(lat, lng, beach_pts), 4)
    return out


def add_anchor_features(df, lat_col="lat_fine", lng_col="lng_fine",
                        offline=False, api_key=None):
    """Add geodesic distances to strategic anchors (navy base, industry, ...)."""
    out = df.copy()
    anchors = _load_anchors(offline=offline, api_key=api_key)
    valid = out[lat_col].notna() & out[lng_col].notna()
    lat = out.loc[valid, lat_col].to_numpy(float)
    lng = out.loc[valid, lng_col].to_numpy(float)
    for name, target in anchors.items():
        col = f"dist_to_{name}_km"
        out[col] = np.nan
        out.loc[valid, col] = np.round(_geodesic_km(lat, lng, target), 4)
    return out


if __name__ == "__main__":
    pois = _load_extended_pois()
    print(pois["category"].value_counts())
    anchors = _load_anchors()
    for k, v in anchors.items():
        print(f"{k:12s} -> {v}")
