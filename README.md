# ✈️ Flight Price & Customer Satisfaction Prediction App

A dual-purpose **machine learning application** that predicts:
- **Flight ticket prices** based on travel details (Regression)
- **Customer satisfaction** based on service ratings and demographics (Classification)

Built with **Python, MLflow**, and **Streamlit**, this end-to-end project includes EDA, model training, evaluation, and a real-time interactive interface.

## Demo ScreenShots:

![Picture](https://github.com/Manav2507/Flight-Price-and-Customer-Satisfaction-Prediction/blob/main/img/website/3_1.png)
![Picture](https://github.com/Manav2507/Flight-Price-and-Customer-Satisfaction-Prediction/blob/main/img/website/3_2.png)
![Picture](https://github.com/Manav2507/Flight-Price-and-Customer-Satisfaction-Prediction/blob/main/img/website/3_3.png)

---

## 🔍 Project Overview

| Module                          | Type        | Goal                                      |
|---------------------------------|-------------|-------------------------------------------|
| 🛫 Flight Price Prediction       | Regression  | Estimate flight ticket prices           |
| 😊 Customer Satisfaction Scoring| Classification | Predict if a customer is satisfied     |

> ✅ Both modules are accessible from separate tabs in a **Streamlit web app**.

---

## 📦 Skills & Tools Used

| Category               | Tools & Libraries                                        |
|------------------------|----------------------------------------------------------|
| Programming            | Python                                                   |
| ML Libraries           | Scikit-learn, XGBoost, RandomForest, Logistic Regression |
| EDA & Visualization    | Pandas, Seaborn, Matplotlib                              |
| ML Tracking            | **MLflow** for experiment tracking                       |
| Deployment             | **Streamlit** Web App                                    |
| Model Serialization    | Pickle (`.pkl`) files                                    |

---

## 🧠 Project 1: Flight Price Prediction

### 🎯 Problem Statement
Predict the **ticket price** of a flight using features like **airline**, **departure/arrival time**, **duration**, and **route**.

### 📈 Business Use Cases
- 💸 Help users plan and budget flight bookings  
- 📊 Support agencies with **price analytics**  
- 🛫 Assist airlines in **dynamic pricing strategies**  

### 🚀 ML Pipeline
1. **Data Cleaning**: Nulls, duplicates, date/time formatting  
2. **Feature Engineering**: Extract duration, stops, time-based features  
3. **Models Used**:
   - Linear Regression (baseline)
   - Random Forest Regressor
   - XGBoost Regressor (Best RMSE)

4. **Evaluation Metrics**:  
   - RMSE  
   - R² Score  

5. **Tracking**: All experiments logged with **MLflow**  
6. **Deployment**: UI for inputting travel details and predicting ticket price

---

## 🧠 Project 2: Customer Satisfaction Prediction

### 🎯 Problem Statement
Classify whether a customer is **Satisfied** or **Dissatisfied** based on flight feedback and service ratings.

### 📈 Business Use Cases
- 🎯 Improve **customer retention**
- 📣 Inform marketing segmentation  
- 🔧 Enhance service delivery & feedback loops

### 🚀 ML Pipeline
1. **Preprocessing**:
   - Label encoding for categorical features  
   - Standardization of numerical fields  

2. **EDA Highlights**:
   - Service quality trends  
   - Age group vs satisfaction

3. **Models Used**:
   - Logistic Regression  
   - Random Forest Classifier  
   - Gradient Boosting Classifier

4. **Evaluation Metrics**:  
   - Accuracy  
   - F1-Score  
   - Confusion Matrix  

5. **Model Management**:  
   - Tracked via **MLflow**  
   - Saved using **Pickle**  

6. **Deployment**: User interface for submitting feedback and getting satisfaction scores

---

## 📂 Dataset Features (Simplified)

| Feature Category    | Sample Features                                |
|---------------------|-------------------------------------------------|
| Flight Details      | Airline, Route, Stops, Duration, Departure Time |
| Customer Info       | Gender, Age, Class, Travel Type                |
| Ratings & Feedback  | Food, Seat Comfort, Cleanliness, Check-in       |
| Target Variables    | Price (regression), Satisfaction (classification) |

---

## 🎮 Streamlit App Features

- Tabbed UI for each module  
- Dropdowns for airline, class, feedback ratings  
- Real-time predictions + model confidence  
- Visual EDA insights (EDA plots, price distributions, satisfaction trends)

---

## 🧪 Model Files

📁 Pre-trained `.pkl` files are available here:  
**[🔗 Download Pickle Models](https://drive.google.com/drive/folders/1WAALLGuIr41FHk8OyWyCRVSOyFebiWev?usp=sharing)**
