# streamlit_app.py
import io
import datetime as _dt
import pandas as pd
import streamlit as st
import pycaret as py
from utils.model_loader import (
    load_yaml_config,
    load_model_from_cfg,
    preprocess_like_training,
)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Melbourne Housing Price — Real-time Prediction",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Melbourne Housing Price — Real-time Prediction")

# ---------- LOAD CONFIG & CHOICES ----------
CFG = load_yaml_config("app_config/config.yaml")
CHOICES = load_yaml_config("app_config/choices.yaml")

# If config has no run_id set, prefill the sidebar with your latest run id
DEFAULT_RUN_ID = "ed23762fdf194456aea08d069eba5541"

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Settings")
use_mlflow = st.sidebar.toggle("Use MLflow model", value=bool(CFG.backend.use_mlflow))
tracking_uri = st.sidebar.text_input("MLflow Tracking URI", value=str(CFG.mlflow.tracking_uri))
_run_id_default = str(CFG.mlflow.run_id or DEFAULT_RUN_ID)
run_id = st.sidebar.text_input("MLflow Run ID", value=_run_id_default)
model_stem = st.sidebar.text_input("Local Model Stem (no .pkl)", value=str(CFG.backend.model_stem))

# ---------- LOAD MODEL (cached) ----------
@st.cache_resource(show_spinner=True)
def _load_model_cached(flag: bool, uri: str, run: str, stem: str):
    class Cfg:
        backend = type("b", (), {"use_mlflow": flag, "model_stem": stem})
        mlflow = type("m", (), {"tracking_uri": uri, "run_id": run})
    return load_model_from_cfg(Cfg, override_use_mlflow=flag, override_run_id=run)

try:
    model = _load_model_cached(use_mlflow, tracking_uri, run_id, model_stem)
    st.sidebar.success(f"Model ready ✅ ({'MLflow' if use_mlflow else 'Local PyCaret'})")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

# ---------- INPUT CONTROLS (dropdowns/sliders/text) ----------
cat = CHOICES.get("categorical", {})
num = CHOICES.get("numeric", {})
txt = CHOICES.get("text", {})

left, right = st.columns(2)

with left:
    suburb = st.selectbox("Suburb", options=cat.get("suburb", ["Richmond"]))

    rooms = st.number_input(
        "Number of Rooms",
        min_value=int(num.get("rooms", {}).get("min", 1)),
        max_value=int(num.get("rooms", {}).get("max", 10)),
        value=int(num.get("rooms", {}).get("default", 3)),
        step=int(num.get("rooms", {}).get("step", 1)),
        help="Total rooms (bedrooms + living spaces).",
    )

    type_ = st.selectbox(
        "Property Type",
        options=cat.get("type", ["h", "u", "t"]),
        help="h=house; u=unit; t=townhouse",
    )

    method = st.selectbox(
        "Sale Method",
        options=cat.get("method", ["S", "SP", "PI", "VB", "PN"]),
        help="S=sold; SP=sold prior; PI=passed in; VB=vendor bid; PN=sold prior (not disclosed)",
    )

    date = st.date_input("Sale Date", value=_dt.date(2023, 3, 10))

    distance = st.number_input(
        "Distance to CBD (km)",
        min_value=float(num.get("distance", {}).get("min", 0.0)),
        max_value=float(num.get("distance", {}).get("max", 50.0)),
        value=float(num.get("distance", {}).get("default", 2.6)),
        step=float(num.get("distance", {}).get("step", 0.1)),
    )

    postcode = st.text_input("Postal Code", value=str(txt.get("postcode_default", "3121")))

with right:
    bedroom2 = st.number_input(
        "Bedrooms (scraped)",
        min_value=int(num.get("bedroom2", {}).get("min", 0)),
        max_value=int(num.get("bedroom2", {}).get("max", 10)),
        value=int(num.get("bedroom2", {}).get("default", 3)),
        step=int(num.get("bedroom2", {}).get("step", 1)),
    )

    bathroom = st.number_input(
        "Bathrooms",
        min_value=int(num.get("bathroom", {}).get("min", 0)),
        max_value=int(num.get("bathroom", {}).get("max", 6)),
        value=int(num.get("bathroom", {}).get("default", 2)),
        step=int(num.get("bathroom", {}).get("step", 1)),
    )

    car = st.number_input(
        "Car Spaces",
        min_value=int(num.get("car", {}).get("min", 0)),
        max_value=int(num.get("car", {}).get("max", 6)),
        value=int(num.get("car", {}).get("default", 1)),
        step=int(num.get("car", {}).get("step", 1)),
    )

    land_size = st.number_input(
        "Land Size (sqm)",
        min_value=int(num.get("land_size", {}).get("min", 0)),
        max_value=int(num.get("land_size", {}).get("max", 2000)),
        value=int(num.get("land_size", {}).get("default", 192)),
        step=int(num.get("land_size", {}).get("step", 1)),
    )

    building_area = st.number_input(
        "Building Area (sqm)",
        min_value=int(num.get("building_area", {}).get("min", 0)),
        max_value=int(num.get("building_area", {}).get("max", 1000)),
        value=int(num.get("building_area", {}).get("default", 111)),
        step=int(num.get("building_area", {}).get("step", 1)),
    )

    year_built = st.number_input(
        "Year Built",
        min_value=int(num.get("year_built", {}).get("min", 1850)),
        max_value=int(num.get("year_built", {}).get("max", 2025)),
        value=int(num.get("year_built", {}).get("default", 1900)),
        step=int(num.get("year_built", {}).get("step", 1)),
    )

    council_area = st.selectbox("Council (Local Government)", options=cat.get("council_area", ["Yarra"]))
    region = st.selectbox("Region", options=cat.get("region", ["Northern Metropolitan"]))

    property_count = st.number_input(
        "Total Properties in Suburb",
        min_value=int(num.get("property_count", {}).get("min", 0)),
        max_value=int(num.get("property_count", {}).get("max", 100000)),
        value=int(num.get("property_count", {}).get("default", 14949)),
        step=int(num.get("property_count", {}).get("step", 1)),
    )

single = pd.DataFrame([{
    "suburb": suburb,
    "rooms": rooms,
    "type": type_,
    "method": method,
    "date": pd.to_datetime(date),
    "distance": distance,
    "postcode": str(postcode),
    "bedroom2": bedroom2,
    "bathroom": bathroom,
    "car": car,
    "land_size": land_size,
    "building_area": building_area,
    "year_built": year_built,
    "council_area": council_area,
    "region": region,
    "property_count": property_count,
}])

# ---------- PREDICT ----------
if st.button("🔮 Predict Price", use_container_width=True):
    try:
        df_pp = preprocess_like_training(single)
        yhat = model.predict(df_pp)
        val = float(yhat.iloc[0] if hasattr(yhat, "iloc") else yhat)
        st.success(f"**Estimated Price:** ${val:,.0f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------- BATCH ----------
st.subheader("📦 Batch Prediction (CSV)")
st.caption("Upload a CSV with the same columns as the form (header row required).")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    try:
        df_in = pd.read_csv(uploaded)
        st.write("Preview:", df_in.head())
        df_pp = preprocess_like_training(df_in)
        preds = model.predict(df_pp)
        out = df_in.copy()
        out["prediction"] = preds.values if hasattr(preds, "values") else preds

        st.success("Batch predictions completed.")
        st.dataframe(out.head(), use_container_width=True)

        buff = io.StringIO()
        out.to_csv(buff, index=False)
        st.download_button("⬇️ Download predictions.csv", buff.getvalue(), file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")

# ---------- FOOTER ----------
st.divider()
st.caption(
    f"Model source: **{'MLflow' if use_mlflow else 'Local PyCaret'}** "
    f"{'(run_id: ' + run_id + ')' if use_mlflow and run_id else ''} · "
    f"Tracking URI: {tracking_uri if use_mlflow else 'N/A'}"
)
