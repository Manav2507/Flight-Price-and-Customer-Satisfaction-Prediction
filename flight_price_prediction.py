import streamlit as st
import pickle
import pandas as pd
from streamlit_option_menu import option_menu
from passenger_satisfaction import satisfaction_form

# ------------------------------
# Load the trained model
# ------------------------------
model = pickle.load(open("./pickle file/flight_rf.pkl", "rb"))

st.set_page_config(page_title="Flight Price Prediction", layout="wide")

# ------------------------------
# Sidebar Navigation
# ------------------------------
coli,colii = st.columns([1,3])
with coli:
    menu = option_menu(
        "Navigation",
        options=["Flight Price Prediction", "Customer Satisfaction"]
    )

with colii:
# ------------------------------
# Flight Price Prediction Section
# ------------------------------
    if menu == "Flight Price Prediction":
        st.title("Flight Price Prediction ✈️")
        st.write("Enter the flight details to get a price estimate")

        # Helper function to extract datetime features
        def convert_to_datetime(date_str):
            date_time = pd.to_datetime(date_str)
            return date_time.day, date_time.month, date_time.hour, date_time.minute

        # Input fields in a 2-column layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Travel Details")
            date_dep = st.date_input("Departure Date")
            time_dep = st.time_input("Departure Time")
            total_stops = st.selectbox("Total Stops", [0, 1, 2, 3])
            airline = st.selectbox("Airline", [
                "Jet Airways", "IndiGo", "Air India", "Multiple carriers",
                "SpiceJet", "Vistara", "GoAir", "Multiple carriers Premium economy",
                "Jet Airways Business", "Vistara Premium economy", "Trujet"
            ])

        with col2:
            st.subheader("Destination and Arrival")
            date_arr = st.date_input("Arrival Date")
            time_arr = st.time_input("Arrival Time")
            source = st.selectbox("Source", ["Delhi", "Kolkata", "Mumbai", "Chennai"])
            destination = st.selectbox("Destination", [
                "Cochin", "Delhi", "New Delhi", "Hyderabad", "Kolkata"
            ])

        # Extract datetime features
        Journey_day, Journey_month, Dep_hour, Dep_min = convert_to_datetime(f"{date_dep} {time_dep}")
        Arrival_hour, Arrival_min = convert_to_datetime(f"{date_arr} {time_arr}")[2:4]

        # Duration calculation
        dur_hour = abs(Arrival_hour - Dep_hour)
        dur_min = abs(Arrival_min - Dep_min)

        # ------------------------------
        # Encoding Categorical Features
        # ------------------------------
        airlines_dict = {
            "Jet Airways": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "IndiGo": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "Air India": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "Multiple carriers": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "SpiceJet": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "Vistara": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "GoAir": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "Multiple carriers Premium economy": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "Jet Airways Business": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            "Vistara Premium economy": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "Trujet": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
        airline_features = airlines_dict.get(airline, [0] * 11)

        source_dict = {
            "Delhi": [1, 0, 0, 0, 0],
            "Kolkata": [0, 1, 0, 0, 0],
            "Mumbai": [0, 0, 1, 0, 0],
            "Chennai": [0, 0, 0, 1, 0],
            "Banglore": [0, 0, 0, 0, 1]
        }
        source_features = source_dict.get(source, [0, 0, 0, 0, 0])

        destination_dict = {
            "Cochin": [1, 0, 0, 0, 0, 0],
            "Delhi": [0, 1, 0, 0, 0, 0],
            "New Delhi": [0, 0, 1, 0, 0, 0],
            "Hyderabad": [0, 0, 0, 1, 0, 0],
            "Kolkata": [0, 0, 0, 0, 1, 0],
            "Banglore": [0, 0, 0, 0, 0, 1]
        }
        destination_features = destination_dict.get(destination, [0, 0, 0, 0, 0, 0])

        # ------------------------------
        # Prediction Button
        # ------------------------------
        st.markdown("<h2 style='text-align: center;'>Predict Flight Price</h2>", unsafe_allow_html=True)

        if st.button("Predict"):
            # Prepare input data
            input_data = [
                total_stops, Journey_day, Journey_month,
                Dep_hour, Dep_min, dur_hour, dur_min
            ] + airline_features + source_features + destination_features

            prediction = model.predict([input_data])

            # Show result
            st.markdown(
                f"<h3 style='text-align: center;'>Predicted Price: ₹ {prediction[0]:,.2f}</h3>",
                unsafe_allow_html=True
            )
    elif menu == "Customer Satisfaction":
        satisfaction_form()