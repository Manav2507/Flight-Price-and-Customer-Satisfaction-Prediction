import streamlit as st
import pickle
import pandas as pd

# Load the model
model = pickle.load(open("flight_rf.pkl", "rb"))

# Function to convert date and time into relevant features
def convert_to_datetime(date_str):
    date_time = pd.to_datetime(date_str)  # Automatically infers the format
    return date_time.day, date_time.month, date_time.hour, date_time.minute

# Streamlit UI
st.title("Flight Price Prediction")

# Input fields in a 2x2 grid layout
col1, col2 = st.columns(2)

# Left Column
with col1:
    st.subheader("Travel Details")
    date_dep = st.date_input("Departure Date")
    time_dep = st.time_input("Departure Time")
    Total_stops = st.selectbox("Total Stops", [0, 1, 2, 3])
    airline = st.selectbox("Airline", [
        "Jet Airways", "IndiGo", "Air India", "Multiple carriers", 
        "SpiceJet", "Vistara", "GoAir", "Multiple carriers Premium economy", 
        "Jet Airways Business", "Vistara Premium economy", "Trujet"
    ])

# Right Column
with col2:
    st.subheader("Destination and Arrival")
    date_arr = st.date_input("Arrival Date")
    time_arr = st.time_input("Arrival Time")
    source = st.selectbox("Source", ["Delhi", "Kolkata", "Mumbai", "Chennai"])
    destination = st.selectbox("Destination", [
        "Cochin", "Delhi", "New Delhi", "Hyderabad", "Kolkata"
    ])

# Convert the datetime values
Journey_day, Journey_month, Dep_hour, Dep_min = convert_to_datetime(f"{date_dep} {time_dep}")
Arrival_hour, Arrival_min = convert_to_datetime(f"{date_arr} {time_arr}")[2:4]

# Duration calculation
dur_hour = abs(Arrival_hour - Dep_hour)
dur_min = abs(Arrival_min - Dep_min)

# ```python
# Airline Encoding
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

# Source Encoding
source_dict = {
    "Delhi": [1, 0, 0, 0],
    "Kolkata": [0, 1, 0, 0],
    "Mumbai": [0, 0, 1, 0],
    "Chennai": [0, 0, 0, 1]
}
source_features = source_dict.get(source, [0, 0, 0, 0])

# Destination Encoding
destination_dict = {
    "Cochin": [1, 0, 0, 0, 0],
    "Delhi": [0, 1, 0, 0, 0],
    "New Delhi": [0, 0, 1, 0, 0],
    "Hyderabad": [0, 0, 0, 1, 0],
    "Kolkata": [0, 0, 0, 0, 1]
}
destination_features = destination_dict.get(destination, [0, 0, 0, 0, 0])

# Centering the Predict button
st.markdown("<h2 style='text-align: center;'>Predict Flight Price</h2>", unsafe_allow_html=True)
if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = [Total_stops, Journey_day, Journey_month, Dep_hour, Dep_min, dur_hour, dur_min] + airline_features + source_features + destination_features
    input_data = [input_data]  # Reshape for the model

    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the prediction result centered
    st.markdown(f"<h3 style='text-align: center;'>Predicted Price: ${prediction[0]:.2f}</h3>", unsafe_allow_html=True)
