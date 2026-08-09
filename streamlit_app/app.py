"""Streamlit app for the Bahrain rent predictor (demo version).

Run from the repository root:

    streamlit run streamlit_app/app.py

Loads the exported joblib artifact (bahrain_rent_model.joblib), lets the user
describe a listing, and shows an estimated monthly rent in BHD.
"""
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


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return joblib.load(ROOT / "bahrain_rent_model.joblib")


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
        "defaults": {
            "governorate": mode(df["Governorate"]),
            "area": mode(df["Area"]),
            "property_type": mode(df["Property_type"]),
            "include_we": mode(df["Include_w_e"]),
            "beds": mode(df["Beds"]),
            "baths": mode(df["Baths"]),
        },
    }


@st.cache_data(show_spinner=False)
def predict_bhd(inputs):
    model = load_model()
    row = pd.DataFrame([dict(inputs)])
    return float(model.predict(row)[0])


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.set_page_config(page_title="Bahrain Rent Predictor", page_icon="🏠",
                   layout="centered")

options = load_options()

st.title("🏠 Bahrain Rent Predictor")
st.caption("Tell us about the property and we'll estimate the monthly rent.")

title = st.text_input(
    "Title",
    placeholder="e.g. Brand new 2 BHK apartment for rent with sea view")

col1, col2 = st.columns(2)
idx = lambda opts, v: opts.index(v) if v in opts else 0  # noqa: E731
with col1:
    governorate = st.selectbox("Governorate", options["governorates"],
                               index=idx(options["governorates"],
                                         options["defaults"]["governorate"]))
    property_type = st.selectbox("Property type", options["property_types"],
                                 index=idx(options["property_types"],
                                           options["defaults"]["property_type"]))
    beds = st.selectbox("Beds", options["beds"],
                        index=idx(options["beds"],
                                  options["defaults"]["beds"]))
    size_sqm = st.number_input("Size (sqm)", min_value=10, max_value=5000,
                               step=5, value=120)
with col2:
    area = st.selectbox("Area", options["areas"],
                        index=idx(options["areas"],
                                  options["defaults"]["area"]))
    include_we = st.selectbox("Rent includes", options["include_we"],
                              index=idx(options["include_we"],
                                        options["defaults"]["include_we"]))
    baths = st.selectbox("Baths", options["baths"],
                         index=idx(options["baths"],
                                   options["defaults"]["baths"]))
    avail_date = st.date_input("Availability date", value=date(2025, 9, 15),
                               min_value=date(2024, 1, 1),
                               max_value=date(2026, 12, 31))

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
    ("Agent_name", ""),
    ("Agency", ""),
    ("Amenities", 8.0),
)

st.divider()

prediction = predict_bhd(inputs)
st.metric("Estimated monthly rent", f"{prediction:,.0f} BHD")
