import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/random_forest_model.pkl")

# Page title
st.title("🏠 California House Price Predictor")

st.write(
    "Enter the house details below to estimate its median value "
    "using a Random Forest machine-learning model."
)

# House details
st.subheader("House Details")

col1, col2 = st.columns(2)

with col1:
    MedInc = st.number_input(
        "Median Income (in $10,000s)",
        value=5.0
    )

    HouseAge = st.number_input(
        "House Age (years)",
        value=20.0
    )

    AveRooms = st.number_input(
        "Average Rooms per House",
        value=5.0
    )

    AveBedrms = st.number_input(
        "Average Bedrooms per House",
        value=1.0
    )


with col2:
    Population = st.number_input(
        "Population in Area",
        value=1000.0
    )

    AveOccup = st.number_input(
        "Average Occupancy (people per household)",
        value=3.0
    )

    Latitude = st.number_input(
        "Latitude",
        value=34.0
    )

    Longitude = st.number_input(
        "Longitude",
        value=-118.0
    )


# Prediction button
if st.button("Predict House Price"):

    house = pd.DataFrame([{
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }])

    prediction = model.predict(house)[0]

    price = prediction * 100000

    st.subheader("Prediction Result")

    st.metric(
        label="Estimated House Value",
        value=f"${price:,.0f}"
    )