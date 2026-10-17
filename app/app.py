import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE = Path(__file__).parent

st.set_page_config(
    page_title="Pakistan Used Car Price Predictor",
    page_icon="🚗"
)


@st.cache_resource
def load_assets():
    model = joblib.load(BASE / "model.joblib")
    options = json.loads((BASE / "options.json").read_text())
    return model, options


def idx(items, value):
    return items.index(value) if value in items else 0


model, opt = load_assets()


st.title("🚗 Pakistan Used Car Price Predictor")
st.caption(
    "Estimate the asking price of a used car in Pakistan using listing details."
)


col1, col2 = st.columns(2)


with col1:
    make = st.selectbox(
        "Make",
        opt["make"],
        index=idx(opt["make"], "Toyota")
    )

    available_models = opt["models_by_make"].get(
        make,
        opt["model"]
    )

    car_model = st.selectbox(
        "Model",
        available_models
    )

    city = st.selectbox(
        "City",
        opt["ad_city"],
        index=idx(opt["ad_city"], "Lahore")
    )

    engine = st.number_input(
        "Engine size (cc)",
        min_value=600,
        max_value=6000,
        value=1300,
        step=100
    )

    mileage = st.number_input(
        "Mileage (km)",
        min_value=100,
        max_value=400000,
        value=80000,
        step=1000
    )


with col2:

    fuel = st.selectbox(
        "Fuel type",
        opt["fuel_type"]
    )

    transmission = st.selectbox(
        "Transmission",
        opt["transmission"]
    )

    assembly = st.selectbox(
        "Assembly",
        opt["assembly"]
    )

    body = st.selectbox(
        "Body type",
        opt["body"]
    )

    color = st.selectbox(
        "Colour",
        opt["color"]
    )

    registered = st.selectbox(
        "Registered in",
        opt["registered"]
    )


if st.button("Predict Price 🚀"):

    row = pd.DataFrame([{
        "assembly": assembly,
        "body": body,
        "ad_city": city,
        "color": color,
        "engine_cc": float(engine),
        "fuel_type": fuel,
        "make": make,
        "mileage": mileage,
        "model": car_model,
        "registered": registered,
        "transmission": transmission
    }])

    row = row[list(model.feature_names_in_)]

    prediction = model.predict(row)[0]

    price = np.expm1(prediction)

    st.success(
        f"Estimated Price: PKR {price:,.0f}"
    )

    st.info(
        f"Realistic range: PKR {price*0.8:,.0f} - "
        f"PKR {price*1.2:,.0f}"
    )


st.divider()

st.caption(
    "Trained on cleaned PakWheels used car listings."
)