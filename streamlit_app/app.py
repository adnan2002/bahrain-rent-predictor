"""Streamlit app for the Bahrain rent predictor.

Run from the repository root:

    streamlit run streamlit_app/app.py

Loads the exported joblib artifact (bahrain_rent_model.joblib), lets the user
describe a listing (inputs sourced from data.csv where the values are a
controlled vocabulary), and predicts the monthly rent in BHD.
"""
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

import joblib
import predictor  # noqa: F401  (registers the pickled class reference)

ARTIFACT = ROOT / "bahrain_rent_model.joblib"
CHECK = ROOT / "export_check.json"
CUSTOM_AGENCY = "— other (type below) —"


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return joblib.load(ARTIFACT)


@st.cache_data(show_spinner=False)
def load_options():
    df = pd.read_csv(ROOT / "data.csv")

    def bed_key(v):
        s = str(v).lower()
        if "studio" in s:
            return (0, 0)
        m = re.search(r"\d+", s)
        return (int(m.group()) if m else 999, 1 if "+" in s else 0)

    def bath_key(v):
        m = re.search(r"\d+", str(v))
        return (int(m.group()) if m else 999, 1 if "+" in str(v) else 0)

    mode = lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None  # noqa: E731
    return {
        "areas": sorted(df["Area"].dropna().unique()),
        "governorates": sorted(df["Governorate"].dropna().unique()),
        "property_types": sorted(df["Property_type"].dropna().unique()),
        "include_we": sorted(df["Include_w_e"].dropna().unique()),
        "beds": sorted(df["Beds"].dropna().unique(), key=bed_key),
        "baths": sorted(df["Baths"].dropna().unique(), key=bath_key),
        "agencies": sorted(df["Agency"].dropna().unique()),
        "examples": df.sample(n=200, random_state=42),
        "defaults": {
            "governorate": mode(df["Governorate"]),
            "area": mode(df["Area"]),
            "property_type": mode(df["Property_type"]),
            "include_we": mode(df["Include_w_e"]),
            "beds": mode(df["Beds"]),
            "baths": mode(df["Baths"]),
            "agency": mode(df["Agency"]),
            "amenities": int(df["Amenities"].median()),
        },
    }


@st.cache_data(show_spinner=False)
def predict_bhd(inputs, model_version="v1"):
    model = load_model()
    row = pd.DataFrame([dict(inputs)])
    return float(model.predict(row)[0])


# --------------------------------------------------------------------------
# Session state helpers
# --------------------------------------------------------------------------
def prefill(row):
    st.session_state["governorate"] = str(row["Governorate"])
    st.session_state["area"] = str(row["Area"])
    st.session_state["property_type"] = str(row["Property_type"])
    st.session_state["include_we"] = str(row["Include_w_e"])
    st.session_state["beds"] = str(row["Beds"])
    st.session_state["baths"] = str(row["Baths"])
    st.session_state["size_sqm"] = float(row["size_sqm"])
    st.session_state["amenities"] = int(row["Amenities"])
    st.session_state["title"] = str(row["Title"])
    st.session_state["agent_name"] = str(row["Agent_name"])
    st.session_state["agency"] = str(row["Agency"])
    st.session_state["avail_date"] = pd.to_datetime(
        row["Availability_date"], format="%d %b %Y", errors="coerce").date()


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.set_page_config(page_title="Bahrain Rent Predictor", page_icon="🏠",
                   layout="wide")

options = load_options()

if "governorate" not in st.session_state:
    defaults = options["defaults"]
    st.session_state["governorate"] = defaults["governorate"]
    st.session_state["area"] = defaults["area"]
    st.session_state["property_type"] = defaults["property_type"]
    st.session_state["include_we"] = defaults["include_we"]
    st.session_state["beds"] = defaults["beds"]
    st.session_state["baths"] = defaults["baths"]
    st.session_state["agency"] = defaults["agency"]
    st.session_state["size_sqm"] = 120
    st.session_state["amenities"] = defaults["amenities"]
    st.session_state["avail_date"] = date(2025, 9, 15)

with st.sidebar:
    st.header("Model")
    if CHECK.exists():
        info = json.loads(CHECK.read_text())
        st.metric("OOF MAE", f"{info.get('oof_mae', '—')}")
        st.metric("Calibration factor", info.get("calibration_factor", "—"))
        weights = info.get("blend_weights", {})
        if weights:
            st.caption("Blend weights: " + ", ".join(
                f"{k} {v * 100:.0f}%" for k, v in weights.items()))
        st.caption(f"Artifact: {info.get('size_mb', '?')} MB")
    st.divider()
    st.caption("Inputs are sourced from data.csv; the model was trained on "
               "propertyfinder.bh listings.")

st.title("Bahrain Rent Predictor")
st.caption("Describe a listing to get a monthly rent estimate in BHD.")

with st.expander("Load an example listing from data.csv"):
    ex = options["examples"].copy()
    ex["size_sqm"] = (ex["Size"].str.extract(r"/\s*([\d,]+)\s*sqm")[0]
                      .str.replace(",", "", regex=False).astype(float))
    ex = ex.dropna(subset=["size_sqm"]).reset_index(drop=True)
    ex["label"] = (ex["Property_id"].astype(str) + " — " + ex["Title"].str[:45])
    picked = st.selectbox("Example", ex["label"].tolist(), key="example_pick",
                          index=None, placeholder="Choose an example…")
    if picked and st.session_state.get("example_applied") != picked:
        prefill(ex[ex["label"] == picked].iloc[0])
        st.session_state["example_applied"] = picked
    st.caption("Picking an example fills the form below — you can still edit "
               "everything.")

col_loc, col_prop = st.columns(2)

with col_loc:
    st.subheader("Location")
    governorate = st.selectbox("Governorate", options["governorates"],
                               key="governorate")
    area = st.selectbox("Area", options["areas"], key="area")

with col_prop:
    st.subheader("Property")
    property_type = st.selectbox("Property type", options["property_types"],
                                 key="property_type")
    include_we = st.selectbox("Rent includes", options["include_we"],
                              key="include_we")
    beds = st.selectbox("Beds", options["beds"], key="beds")
    baths = st.selectbox("Baths", options["baths"], key="baths")
    size_sqm = st.number_input("Size (sqm)", min_value=10, max_value=5000,
                               step=5, key="size_sqm")
    amenities = st.number_input("Number of amenities", min_value=0,
                                max_value=60, key="amenities")

st.subheader("Listing")
title = st.text_input("Title",
                      placeholder="e.g. Brand new 2 BHK apartment for rent in "
                                  "Al Juffair with sea view",
                      key="title")
col_agent, col_agency = st.columns(2)
with col_agent:
    agent_name = st.text_input("Agent name",
                               placeholder="e.g. Savio Fernandes",
                               key="agent_name")
with col_agency:
    agency_options = [CUSTOM_AGENCY] + options["agencies"]
    agency = st.selectbox("Agency", agency_options, key="agency")
    if agency == CUSTOM_AGENCY:
        agency = st.text_input("Custom agency", key="agency_custom")

col_date, _ = st.columns(2)
with col_date:
    avail_date = st.date_input("Availability date", key="avail_date",
                               min_value=date(2024, 1, 1),
                               max_value=date(2026, 12, 31))

missing = [label for label, val in
           [("title", title), ("agent name", agent_name)]
           if not val.strip()]
if missing:
    st.warning("Missing inputs: " + ", ".join(missing) + ". "
               "The prediction will use the global market average for "
               "unknown values.")

inputs = (
    ("Property_type", property_type),
    ("Include_w_e", include_we),
    ("Title", title.strip()),
    ("Area", area),
    ("Governorate", governorate),
    ("Beds", beds),
    ("Baths", baths),
    ("Size", f"{round(size_sqm * 10.7639):,} sqft / {size_sqm:,.0f} sqm"),
    ("Availability_date", avail_date.strftime("%d %b %Y")),
    ("Agent_name", agent_name.strip()),
    ("Agency", agency.strip()),
    ("Amenities", float(amenities)),
)

prediction = predict_bhd(inputs)

st.divider()
col_metric, col_note = st.columns([1, 2])
with col_metric:
    st.metric("Predicted monthly rent", f"{prediction:,.0f} BHD",
              delta=None)
with col_note:
    st.caption("Estimate from the bagged ensemble (4 learners × 3 seeds) "
               "blended on out-of-fold predictions (OOF MAE ≈ 85 BHD). "
               "Listings predicted above 900 BHD are scaled by the "
               "premium-segment calibration factor. Rents are capped at 0.")
