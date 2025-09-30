import streamlit as st
import numpy as np
import pickle

# Load satisfaction model


def satisfaction_form():
    st.title("Customer Satisfaction Prediction 😊")
    st.write("Fill in the passenger details to predict satisfaction")

    with open("./pickle file/Customer_Satisfaction.pkl", "rb") as file:
        satisfaction_model = pickle.load(file)

    with st.form(key="satisfaction_form"):
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

        Gender_map = {"Female": 0, "Male": 1}
        CustomerType_map = {"Loyal Customer": 1, "Disloyal Customer": 0}
        TypeofTravel_map = {"Personal Travel": 0, "Business Travel": 1}
        Class_map = {"Eco": [1, 0], "Eco Plus": [0, 1], "Business": [0, 0]}

        class_dummies = Class_map[Class]
        satisfaction_input = [
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
            *class_dummies
        ]

        submitted = st.form_submit_button("Predict Satisfaction")
        if submitted:
            satisfaction_input_np = np.array(satisfaction_input).reshape(1, -1)
            satisfaction_pred = satisfaction_model.predict(satisfaction_input_np)
            if satisfaction_pred[0] == 0:
                st.error("Prediction: Dissatisfied ❌")
            else:
                st.success("Prediction: Satisfied ✅")