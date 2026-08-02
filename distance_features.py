"""
Distance-to-POI feature engineering for Bahrain property data.

Computes, for any (latitude, longitude) in Bahrain:
  - dist_to_coastline_km                  (nearest OpenStreetMap coastline)
  - dist_to_manama_cbd_km                 (Bab Al Bahrain / Government Ave anchor)
  - dist_to_airport_km                    (Bahrain International Airport)
  - dist_to_causeway_km                   (King Fahd Causeway entry, Al Jasra toll)
  - dist_to_nearest_international_school_km
  - dist_to_nearest_mall_km

Usage:
    from distance_features import add_distance_features
    df = add_distance_features(df, lat_col="latitude", lng_col="longitude")

Distances are geodesic/projected (UTM-39N), in kilometres. Always >= 0;
sign/direction of the effect is left to the model.
"""
import json
import os

import numpy as np
import pandas as pd
import requests
from pyproj import Geod, Transformer
from shapely import (
    LineString, STRtree, distance as shp_distance,
    from_wkt, points as shp_points,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
POIS_CSV = os.path.join(_HERE, "bahrain_pois.csv")
COASTLINE_GEOJSON = os.path.join(_HERE, "bahrain_coastline.geojson")

# Anchor points (WGS84)
MANAMA_CBD = (26.2339813, 50.5756748)      # Bab Al Bahrain / Government Ave
AIRPORT = (26.2671847, 50.6302827)         # Bahrain International Airport
CAUSEWAY_ENTRY = (26.172373, 50.4577019)   # Al Jasra toll booth (Bahrain side)

# UTM zone 39N covers Bahrain (48E-54E); distances in metres
_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)
_GEOD = Geod(ellps="WGS84")

# Overpass fallback if cached coastline file is missing
_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_OVERPASS_QUERY = ('[out:json][timeout:60];'
                   'way["natural"="coastline"](25.40,50.15,26.70,50.95);'
                   'out geom;')

_COAST_TREE = None   # cached STRtree of coastline segments (UTM)


def _fetch_and_cache_coastline(path=COASTLINE_GEOJSON):
    """Fetch Bahrain coastline ways from Overpass and cache as GeoJSON."""
    r = requests.get(_OVERPASS_URL, params={"data": _OVERPASS_QUERY},
                     timeout=180, headers={"User-Agent": "distance-features/1.0"})
    r.raise_for_status()
    ways = [
        [[pt["lon"], pt["lat"]] for pt in el["geometry"]]
        for el in r.json().get("elements", [])
        if el.get("type") == "way" and len(el.get("geometry", [])) >= 2
    ]
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {},
         "geometry": {"type": "LineString", "coordinates": w}} for w in ways
    ]}
    with open(path, "w") as f:
        json.dump(fc, f)
    return path


def _load_coastline_tree(path=COASTLINE_GEOJSON):
    """Load coastline, split into segments, project to UTM, build STRtree."""
    global _COAST_TREE
    if _COAST_TREE is not None:
        return _COAST_TREE
    if not os.path.exists(path):
        _fetch_and_cache_coastline(path)
    with open(path) as f:
        fc = json.load(f)

    lons, lats, cuts = [], [], [0]
    for feat in fc["features"]:
        coords = feat["geometry"]["coordinates"]
        lons.extend(c[0] for c in coords)
        lats.extend(c[1] for c in coords)
        cuts.append(len(coords))
    xs, ys = _TO_UTM.transform(np.array(lons), np.array(lats))

    segments = []
    pos = 0
    for n in cuts[1:]:
        way_pts = np.column_stack([xs[pos:pos + n], ys[pos:pos + n]])
        for i in range(n - 1):
            segments.append(LineString([way_pts[i], way_pts[i + 1]]))
        pos += n
    _COAST_TREE = STRtree(np.array(segments, dtype=object))
    return _COAST_TREE


def _to_utm_xy(lat, lng):
    return _TO_UTM.transform(np.asarray(lng, dtype=float),
                             np.asarray(lat, dtype=float))


def _geodesic_km(lat, lng, target_lat_lng):
    """Vectorized WGS84 geodesic distance to a fixed point, in km."""
    tlat, tlng = target_lat_lng
    _, _, m = _GEOD.inv(np.full_like(lng, tlng), np.full_like(lat, tlat),
                        np.asarray(lng, float), np.asarray(lat, float))
    return m / 1000.0


def _min_geodesic_km(lat, lng, points_lat_lng):
    """Vectorized minimum geodesic distance to a set of points, in km."""
    pts = np.asarray(points_lat_lng, dtype=float)
    n, m_pts = len(lat), len(pts)
    lat2 = np.broadcast_to(np.asarray(lat, float)[:, None], (n, m_pts))
    lng2 = np.broadcast_to(np.asarray(lng, float)[:, None], (n, m_pts))
    tlat = np.broadcast_to(pts[:, 0][None, :], (n, m_pts))
    tlng = np.broadcast_to(pts[:, 1][None, :], (n, m_pts))
    _, _, m = _GEOD.inv(tlng, tlat, lng2, lat2)   # (N, M)
    return m.min(axis=1) / 1000.0


def _coastline_km(lat, lng):
    """Vectorized distance to nearest coastline segment (UTM projection), km."""
    tree = _load_coastline_tree()
    xs, ys = _to_utm_xy(lat, lng)
    pts = shp_points(xs, ys)
    idx = tree.query_nearest(pts, all_matches=False)[1]  # (2, N) -> tree indices
    nearest_segments = tree.geometries[idx]
    return shp_distance(pts, nearest_segments) / 1000.0


def _load_pois(path=POIS_CSV):
    df = pd.read_csv(path)
    schools = df.loc[df["category"] == "international_school",
                     ["latitude", "longitude"]].values
    malls = df.loc[df["category"] == "mall", ["latitude", "longitude"]].values
    return schools, malls


def add_distance_features(df, lat_col="latitude", lng_col="longitude",
                          pois_path=POIS_CSV):
    """Return a copy of df with the six dist_*_km columns added."""
    out = df.copy()
    valid = out[lat_col].notna() & out[lng_col].notna()
    lat = out.loc[valid, lat_col].to_numpy(float)
    lng = out.loc[valid, lng_col].to_numpy(float)

    schools, malls = _load_pois(pois_path)
    feats = {
        "dist_to_coastline_km": _coastline_km(lat, lng),
        "dist_to_manama_cbd_km": _geodesic_km(lat, lng, MANAMA_CBD),
        "dist_to_airport_km": _geodesic_km(lat, lng, AIRPORT),
        "dist_to_causeway_km": _geodesic_km(lat, lng, CAUSEWAY_ENTRY),
        "dist_to_nearest_international_school_km": _min_geodesic_km(lat, lng, schools),
        "dist_to_nearest_mall_km": _min_geodesic_km(lat, lng, malls),
    }
    for name, vals in feats.items():
        out[name] = np.nan
        out.loc[valid, name] = np.round(vals, 4)
    return out


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(_HERE, "data.csv"))
    coords = pd.read_csv(os.path.join(_HERE, "area_coordinates.csv"))
    df["latitude"] = df["Area"].map(coords.set_index("Area")["latitude"])
    df["longitude"] = df["Area"].map(coords.set_index("Area")["longitude"])
    df = add_distance_features(df)
    out_path = os.path.join(_HERE, "data_with_distances.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows)")
