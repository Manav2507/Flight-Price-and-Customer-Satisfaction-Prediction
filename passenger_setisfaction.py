import streamlit as st
import pickle
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
with open("Customer_Satisfaction.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Airline Satisfaction Prediction")

# Input fields in a grid layout
col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ["Female", "Male"])
    CustomerType = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
    Age = st.number_input("Age", min_value=1, max_value=100, value=25, step=1)
    TypeofTravel = st.selectbox("Type of Travel", ["Personal Travel", "Business Travel"])
    Class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
    FlightDistance = st.number_input("Flight Distance (in km)", min_value=0, step=1)
    InflightWifiService = st.selectbox("Inflight Wifi Service (1-5)", [1, 2, 3, 4, 5])
    DepartureArrivalTimeConvenient = st.selectbox("Departure/Arrival Time Convenient (1-5)", [1, 2, 3, 4, 5])
    EaseofOnlineBooking = st.selectbox("Ease of Online Booking (1-5)", [1, 2, 3, 4, 5])
    GateLocation = st.selectbox("Gate Location (1-5)", [1, 2, 3, 4, 5])
    FoodandDrink = st.selectbox("Food and Drink (1-5)", [1, 2, 3, 4, 5])

with col2:
    
    OnlineBoarding = st.selectbox("Online Boarding (1-5)", [1, 2, 3, 4, 5])
    BaggageHandling = st.selectbox("Baggage Handling (1-5)", [1, 2, 3, 4, 5])
    CheckinService = st.selectbox("Check-in Service (1-5)", [1, 2, 3, 4, 5])
    SeatComfort = st.selectbox("Seat Comfort (1-5)", [1, 2, 3, 4, 5])
    InflightEntertainment = st.selectbox("Inflight Entertainment (1-5)", [1, 2, 3, 4, 5])
    OnboardService = st.selectbox("On-board Service (1-5)", [1, 2, 3, 4, 5])
    LegroomService = st.selectbox("Legroom Service (1-5)", [1, 2, 3, 4, 5])
    InflightService = st.selectbox("Inflight Service (1-5)", [1, 2, 3, 4, 5])
    Cleanliness = st.selectbox("Cleanliness (1-5)", [1, 2, 3, 4, 5])
    DepartureDelayInMinutes = st.number_input("Departure Delay (in minutes)", min_value=0, step=1)
    ArrivalDelayInMinutes = st.number_input("Arrival Delay (in minutes)", min_value=0, step=1)
    # DepartureDelayInMinutes = st.slider("Departure Delay (in minutes)", min_value=0, max_value=300, value=0, step=1)
    # ArrivalDelayInMinutes = st.slider("Arrival Delay (in minutes)", min_value=0, max_value=300, value=0, step=1 )

    
# Map categorical values to numeric values
Gender_map = {"Female": 0, "Male": 1}
CustomerType_map = {"Loyal Customer": 1, "Disloyal Customer": 0}
TypeofTravel_map = {"Personal Travel": 0, "Business Travel": 1}
Class_map = {"Eco": [1, 0], "Eco Plus": [0, 1], "Business": [0, 0]}  # Dummy variables for class

# Convert inputs to numerical format
class_dummies = Class_map[Class]
input_data = [
    Gender_map[Gender],
    CustomerType_map[CustomerType],
    Age,
    TypeofTravel_map[TypeofTravel],
    FlightDistance,
    InflightWifiService,
    DepartureArrivalTimeConvenient,
    EaseofOnlineBooking,
    GateLocation,
    FoodandDrink,
    OnlineBoarding,
    SeatComfort,
    InflightEntertainment,
    OnboardService,
    LegroomService,
    BaggageHandling,
    CheckinService,
    InflightService,
    Cleanliness,
    DepartureDelayInMinutes,
    ArrivalDelayInMinutes,
    *class_dummies  # Add dummy variables for "Class"
]

# Predict satisfaction level
if st.button("Predict Satisfaction"):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("Prediction: Neutral or Dissatisfied")
    else:
        st.success("Prediction: Satisfied")
