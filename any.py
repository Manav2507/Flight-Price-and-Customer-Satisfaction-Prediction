import pandas as pd
import numpy as np
import datetime as dt
import math
train_data = pd.read_csv("Flight_Price.csv")
train_data.head(15)
## Data Preprocessing
train_data["Journey_Day"] = pd.to_datetime(train_data["Date_of_Journey"], format="%d/%m/%Y").dt.day

train_data["Journey_Month"] = pd.to_datetime(train_data["Date_of_Journey"], format="%d/%m/%Y").dt.month
train_data.head()
train_data.drop(["Date_of_Journey"], axis=1, inplace=True)
train_data["Dep_hour"] = pd.to_datetime(train_data["Dep_Time"], format='%H:%M').dt.hour

train_data["Dep_minute"] = pd.to_datetime(train_data["Dep_Time"], format='%H:%M').dt.minute
train_data.drop(["Dep_Time"], axis=1, inplace=True)
# Step 1: Split the time from the full datetime string
train_data["Arrival_Time_Cleaned"] = train_data["Arrival_Time"].str.split().str[-1]

# Step 2: Parse the cleaned time strings to extract hour and minute
train_data["Arrival_hour"] = pd.to_datetime(train_data["Arrival_Time_Cleaned"], format='%H:%M').dt.hour
train_data["Arrival_minute"] = pd.to_datetime(train_data["Arrival_Time_Cleaned"], format='%H:%M').dt.minute

train_data.drop(columns=["Arrival_Time_Cleaned","Arrival_Time"], axis=1, inplace=True)
duration = list(train_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h " + duration[i]

duration_hour = []
duration_minute = []

for i in range(len(duration)):
    duration_hour.append(int(duration[i].split(sep = 'h')[0]))
    duration_minute.append(int(duration[i].split(sep = 'm')[0].split()[-1]))
train_data["duration_hour"] = duration_hour
train_data["duration_minute"] = duration_minute
train_data.drop(["Duration"], axis=1, inplace=True)
## Handling Categorical Data
import seaborn as sns
import matplotlib.pyplot as plt
sns.catplot(y = "Price", x = "Airline", data = train_data.sort_values("Price", ascending=False), height=6, aspect=3, kind="boxen")
plt.show()
Airline = pd.get_dummies(train_data["Airline"], drop_first=True, dtype=int, prefix="Airline")
Airline.head()
Source = pd.get_dummies(train_data["Source"], drop_first=True, dtype=int, prefix="Source")
Source.head()
Destination = pd.get_dummies(train_data["Destination"], drop_first=True, dtype=int, prefix="Destination")
Destination.head()
train_data["Total_Stops"].value_counts()
train_data.replace({"1 stop": 1, "non-stop": 0, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace=True)

train_data.head()
train_data.drop(columns=["Additional_Info","Route"], axis=1, inplace=True)
data_train = pd.concat([train_data, Airline, Source, Destination], axis=1)
data_train["Total_Stops"] = data_train["Total_Stops"].fillna(0).astype(int)
data_train.drop(columns=["Airline","Source","Destination"], axis=1, inplace=True)
data_train.head()
## Test data
import pandas as pd
import numpy as np
import datetime as dt
import math
test_data = pd.read_csv("Flight_Price.csv")
test_data.head(15)
## Data Preprocessing
test_data["Journey_Day"] = pd.to_datetime(test_data["Date_of_Journey"], format="%d/%m/%Y").dt.day

test_data["Journey_Month"] = pd.to_datetime(test_data["Date_of_Journey"], format="%d/%m/%Y").dt.month
test_data.head()
test_data.drop(["Date_of_Journey"], axis=1, inplace=True)
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"], format='%H:%M').dt.hour

test_data["Dep_minute"] = pd.to_datetime(test_data["Dep_Time"], format='%H:%M').dt.minute
test_data.drop(["Dep_Time"], axis=1, inplace=True)
# Step 1: Split the time from the full datetime string
test_data["Arrival_Time_Cleaned"] = test_data["Arrival_Time"].str.split().str[-1]

# Step 2: Parse the cleaned time strings to extract hour and minute
test_data["Arrival_hour"] = pd.to_datetime(test_data["Arrival_Time_Cleaned"], format='%H:%M').dt.hour
test_data["Arrival_minute"] = pd.to_datetime(test_data["Arrival_Time_Cleaned"], format='%H:%M').dt.minute

test_data.drop(columns=["Arrival_Time_Cleaned","Arrival_Time"], axis=1, inplace=True)
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h " + duration[i]

duration_hour = []
duration_minute = []

for i in range(len(duration)):
    duration_hour.append(int(duration[i].split(sep = 'h')[0]))
    duration_minute.append(int(duration[i].split(sep = 'm')[0].split()[-1]))
test_data["duration_hour"] = duration_hour
test_data["duration_minute"] = duration_minute
test_data.drop(["Duration"], axis=1, inplace=True)
## Handling Categorical Data
import seaborn as sns
import matplotlib.pyplot as plt
sns.catplot(y = "Price", x = "Airline", data = test_data.sort_values("Price", ascending=False), height=6, aspect=3, kind="boxen")
plt.show()
Airline = pd.get_dummies(test_data["Airline"], drop_first=True, dtype=int)
Airline.head()
Source = pd.get_dummies(test_data["Source"], drop_first=True, dtype=int)
Source.head()
Destination = pd.get_dummies(test_data["Destination"], drop_first=True, dtype=int)
Destination.head()
test_data["Total_Stops"].value_counts()
test_data.replace({"1 stop": 1, "non-stop": 0, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace=True)

test_data.head()
test_data.drop(columns=["Additional_Info","Route"], axis=1, inplace=True)
data_test = pd.concat([test_data, Airline, Source, Destination], axis=1)
data_test["Total_Stops"] = data_test["Total_Stops"].fillna(0).astype(int)
data_test.drop(columns=["Airline","Source","Destination"], axis=1, inplace=True)
data_test.head()
## Train and Test data
data_train.columns
X = data_train.loc[:,['Total_Stops','Journey_Day', 'Journey_Month', 'Dep_hour',
       'Dep_minute', 'Arrival_hour', 'Arrival_minute', 'duration_hour',
       'duration_minute', 'Airline_Air India', 'Airline_GoAir',
       'Airline_IndiGo', 'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()
y = data_train.iloc[:,1]
y.head()
# Finds correlation between Independent and dependent attributes

plt.figure(figsize = (18,18))
sns.heatmap(data_train.corr(), annot = True, cmap = "RdYlGn")

plt.show()
# Important feature using ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)
print(selection.feature_importances_)
plt.figure(figsize=(12,8))
feat_importance = pd.Series(selection.feature_importances_,index=X.columns)
feat_importance.nlargest(20).plot(kind='barh')
plt.show()
## Fitting model using Random Forest
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)
from sklearn.ensemble import RandomForestRegressor

reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)
y_pred = reg_rf.predict(X_test)
reg_rf.score(X_train, y_train)
reg_rf.score(X_test,y_test)
plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
metrics.r2_score(y_test,y_pred)
## Hyperparameter Tuning

from sklearn.model_selection import RandomizedSearchCV
#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]    
# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
prediction = rf_random.predict(X_test)
plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("Prediction")
plt.show()
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
import pickle
# open a file, where you ant to store the data
file = open('flight_rf.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)
model = open('flight_rf.pkl','rb')

forest = pickle.load(model)
y_prediction = forest.predict(X_test)
metrics.r2_score(y_test, prediction)
## Linear Regression
from sklearn.linear_model import LinearRegression
lm =LinearRegression()
lm.fit(X_train,y_train)
prediction_LR = lm.predict(X_test)

prediction_LR
plt.scatter(y_test,prediction_LR)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
## KNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import zscore
X = data_train.loc[:,['Total_Stops','Journey_Day', 'Journey_Month', 'Dep_hour',
       'Dep_minute', 'Arrival_hour', 'Arrival_minute', 'duration_hour',
       'duration_minute', 'Airline_Air India', 'Airline_GoAir',
       'Airline_IndiGo', 'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
y = data_train.iloc[:,1]
XScaled  = X.apply(zscore)
X_train, X_test, y_train, y_test = train_test_split(XScaled, y, test_size=0.25, random_state=1)
NNH = KNeighborsRegressor(n_neighbors= 5 , weights = 'distance' )
NNH.fit(X_train, y_train)
predicted_labels_train = NNH.predict(X_train)
NNH.score(X_train, y_train)
predicted_labels_test = NNH.predict(X_test)

NNH.score(X_test, y_test)
from sklearn import metrics

mae = metrics.mean_absolute_error(y_train, predicted_labels_train)
mse = metrics.mean_squared_error(y_train, predicted_labels_train)
r2 = metrics.r2_score(y_train, predicted_labels_train)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R² Score: {r2}")

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Initialize the KNN regressor
knn = KNeighborsRegressor()

# Define the hyperparameter grid
param_grid = {
    "n_neighbors": [2, 3, 5, 7, 10, 15],
    "weights": ["uniform", "distance"],
    "p": [1, 2]  # Manhattan and Euclidean distances
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=5,
    verbose=2,
    n_jobs=-1
)

# Fit the model to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and performance
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("Best Parameters:", best_params)
print("RMSE:", rmse)

## MLflow
import mlflow
mlflow.set_experiment("1st_Random_Forest")
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
## Random Forest
with mlflow.start_run(run_name="RandomizedSearchCV"):
    rf_random = RandomizedSearchCV(
        estimator=reg_rf,
        param_distributions=random_grid,
        scoring="neg_mean_squared_error",
        n_iter=10,
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    rf_random.fit(X_train, y_train)
    
    # Log best parameters and performance
    best_params = rf_random.best_params_
    mlflow.log_params(best_params)
    
    prediction = rf_random.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, prediction)
    mse = metrics.mean_squared_error(y_test, prediction)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, prediction)
    
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    
    # Log the best model
    mlflow.sklearn.log_model(rf_random.best_estimator_, "best_random_forest_model")

## Linear Regression
import mlflow
import mlflow.sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile

# Start an MLflow run
with mlflow.start_run(run_name="Linear Regression Model"):
    # Train and predict
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    prediction_LR = lm.predict(X_test)

    # Calculate evaluation metrics
    mae = metrics.mean_absolute_error(y_test, prediction_LR)
    mse = metrics.mean_squared_error(y_test, prediction_LR)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, prediction_LR)

    # Log metrics to MLflow
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Log the model
    mlflow.sklearn.log_model(lm, "linear_regression_model")
## KNN



import pandas as pd
import numpy as np
train_data = pd.read_csv("Passenger_Satisfaction.csv")
train_data.info()
train_data.drop(columns=["Unnamed: 0","id"], axis=1, inplace=True)
train_data.head()
train_data.Gender.replace({"Male":1, "Female":0}, inplace=True)
train_data["Customer Type"].replace({"Loyal Customer":1, "disloyal Customer":0}, inplace=True)
train_data["Type of Travel"].replace({"Personal Travel":1, "Business travel":0}, inplace=True)
train_data = train_data.join(pd.get_dummies(train_data["Class"], drop_first=True, dtype=int, prefix="class"))
train_data.drop(columns=["Class"], axis=1, inplace=True)
train_data.isnull().sum()
train_data['Arrival Delay in Minutes'].fillna(train_data['Arrival Delay in Minutes'].mean(), inplace=True)
mapping = {'neutral or dissatisfied': 0, 'satisfied': 1}
train_data['satisfaction'] = train_data['satisfaction'].map(mapping)

data_train = train_data
data_train.head()
import pandas as pd
import numpy as np
test_data = pd.read_csv("Passenger_Satisfaction.csv")
test_data.info()
test_data.drop(columns=["Unnamed: 0","id"], axis=1, inplace=True)
test_data.head()
test_data.Gender.replace({"Male":1, "Female":0}, inplace=True)
test_data["Customer Type"].replace({"Loyal Customer":1, "disloyal Customer":0}, inplace=True)
test_data["Type of Travel"].replace({"Personal Travel":1, "Business travel":0}, inplace=True)
test_data = test_data.join(pd.get_dummies(test_data["Class"], drop_first=True, dtype=int, prefix="class"))
test_data.drop(columns=["Class"], axis=1, inplace=True)
test_data.isnull().sum()
test_data['Arrival Delay in Minutes'].fillna(test_data['Arrival Delay in Minutes'].mean(), inplace=True)
mapping = {'neutral or dissatisfied': 0, 'satisfied': 1}
test_data['satisfaction'] = test_data['satisfaction'].map(mapping)
data_test = test_data
data_test.head()
data_train.columns
X = data_train.loc[:,['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Flight Distance',
       'Inflight wifi service', 'Departure/Arrival time convenient',
       'Ease of Online booking', 'Gate location', 'Food and drink',
       'Online boarding', 'Seat comfort', 'Inflight entertainment',
       'On-board service', 'Leg room service', 'Baggage handling',
       'Checkin service', 'Inflight service', 'Cleanliness',
       'Departure Delay in Minutes', 'Arrival Delay in Minutes',
        'class_Eco', 'class_Eco Plus']]
y = data_train.iloc[:,-3]
y.head()
# Finds correlation between Independent and dependent attributes
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (18,18))
sns.heatmap(data_train.corr(), annot = True, cmap = "RdYlGn")

plt.show()
from sklearn.ensemble import ExtraTreesClassifier
selection = ExtraTreesClassifier()
selection.fit(X,y)
plt.figure(figsize=(12,8))
feat_importance = pd.Series(selection.feature_importances_,index=X.columns)
feat_importance.nlargest(20).plot(kind='barh')
plt.show()
## Fitting model using Random Forest
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=42)
from sklearn.ensemble import RandomForestClassifier

reg_cl = RandomForestClassifier(n_estimators=501)
reg_cl.fit(X_train, y_train)
reg_cl.score(X_train, y_train)
reg_cl.score(X_test, y_test)
ytrain_predict = reg_cl.predict(X_train)
ytest_predict = reg_cl.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_train,ytrain_predict)
confusion_matrix(y_test,ytest_predict)
print(classification_report(y_train,ytrain_predict))
print(classification_report(y_test,ytest_predict))
data_train["satisfaction"].value_counts(normalize=True)
## Logistic Regression
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
model_lr.fit(X_train,y_train)
trainpredict=model_lr.predict(X_train)
testpredict=model_lr.predict(X_test)
model_lr.score(X_train, y_train)
model_lr.score(X_test, y_test)
from sklearn import metrics

metrics.confusion_matrix(y_train,trainpredict)
metrics.confusion_matrix(y_test,testpredict)
print(metrics.classification_report(y_train,trainpredict))
print(metrics.classification_report(y_test,testpredict))
## Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()
dt_model = DecisionTreeClassifier(criterion = 'entropy' )
dt_model.fit(X_train, y_train)
ytrain_predict = dt_model.predict(X_train)
ytest_predict = dt_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_train, ytrain_predict))
print(classification_report(y_test, ytest_predict))
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid={
    'max_depth': [4,5,6,7],
    'min_samples_leaf': [1,2,3,4],
    'min_samples_split': [10,20,30,40],
    'max_features':[14,15,16,17],
    'criterion':['gini','entropy']
}

grid_search=GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5,scoring='accuracy')
grid_search.fit(X_train,y_train)
grid_search.best_params_
ytrain_predict = grid_search.predict(X_train)
ytest_predict = grid_search.predict(X_test)
print(classification_report(y_train, ytrain_predict))
print(classification_report(y_test, ytest_predict))


## Naive Bayes Model
# Import Naive Bayes library and Metrics to build the model, confusion metrics and classification report

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
NB_model = GaussianNB()
NB_model.fit(X_train, y_train)
y_train_predict = NB_model.predict(X_train)
model_score = NB_model.score(X_train, y_train)
print(model_score)
print(metrics.confusion_matrix(y_train, y_train_predict))
print(metrics.classification_report(y_train, y_train_predict))
y_test_predict = NB_model.predict(X_test)
model_score = NB_model.score(X_test, y_test)
print(model_score)
print(metrics.confusion_matrix(y_test, y_test_predict))
print(metrics.classification_report(y_test, y_test_predict))
## KNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import zscore
XScaled  = X.apply(zscore)   # convert all attributes to Z scale
NNH = KNeighborsClassifier(n_neighbors= 5 , weights = 'distance' )
NNH.fit(X_train, y_train)
predicted_labels_train = NNH.predict(X_train)
NNH.score(X_train, y_train)
predicted_labels_test = NNH.predict(X_test)
NNH.score(X_test, y_test)
metrics.confusion_matrix(y_train,predicted_labels_train)
metrics.confusion_matrix(y_test,predicted_labels_test)
print(metrics.classification_report(y_train,predicted_labels_train,digits=3))
print("-------------------------------------------------------")
print(metrics.classification_report(y_test,predicted_labels_test,digits=3))
## Gradient Boosting
# import GB and build the model on training data

from sklearn.ensemble import GradientBoostingClassifier
gbcl = GradientBoostingClassifier(random_state=1)
gbcl = gbcl.fit(X_train, y_train)
## Performance Matrix on train data set
y_train_predict = gbcl.predict(X_train)
model_score = gbcl.score(X_train, y_train)
print("Train : ",model_score)
print("Train : ",metrics.confusion_matrix(y_train, y_train_predict))
print("Train : ",metrics.classification_report(y_train, y_train_predict))
## Performance Matrix on test data set
y_test_predict = gbcl.predict(X_test)
model_score = gbcl.score(X_test, y_test)
print("Test : ",model_score)
print("Test : ",metrics.confusion_matrix(y_test, y_test_predict))
print("Test : ",metrics.classification_report(y_test, y_test_predict))
import pickle

# Save the fully trained model
with open('Customer_Satisfaction.pkl', 'wb') as file:
    pickle.dump(reg_cl, file)
## MLflow

