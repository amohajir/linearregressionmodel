import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, linear_model

# Step 1

# Reading in the data file, skipping the first row containing the column names, and naming the columns
df = pd.read_csv('e-scooter-trips-2019.csv', names = ['date','hour', 'am-pm', 'distance', 'trip-duration'], skiprows = 1)
#df

# Dropping the date column and extracting the hour from the 'hour' column
df = df.drop(['date'], axis=1) # axis = 1 represents columns
df['hour'] = pd.to_datetime(df['hour'], format='%H:%M:%S').dt.hour
#df

# Step 2

# AM changed to 0 and PM changed to 1
le = preprocessing.LabelEncoder()
df['am-pm'] = le.fit_transform(df['am-pm'])
#df

# Step 3

# Rows with distance greater than 0 meters and less than or equal to 30,000 meters
df = df[df['distance'] > 0]
df = df[df['distance'] <= 30000]
#df

# Step 4

# Rows with trip-duration > 100 seconds; dataframe is now cleaned up
df = df[df['trip-duration'] > 100]
#df

# Building a 3-variable linear regression model to predict the fourth with the cleaned-up dataframe
# Y = a*X1 + b*X2 + c*X3, where the letters represent constants estimated by sklearn

X = df[['hour', 'am-pm', 'distance']] # X represents the variables used to predict Y
Y = df['trip-duration'] # Y represents our predicted target variable

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) # random_state ensures that the generated splits are reproducible, 0 is the seed that we provide to the random number generator so that our executed code always produces the same results

escooter_model = linear_model.LinearRegression()
escooter_model.fit(X_train, Y_train)

# Step 5

# Model Prediction
escooter_model.predict([[9, 0, 1000]]) # 9 AM, 1000 m
#predicted trip-duration: 411.82617527

# If extra [] are not used...
#ValueError: Expected 2D array, got 1D array instead:
#array=[   9    0 1000].
#Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

#escooter_model.predict([[1, 1, 1000]]) # 1 PM, 1000 m
#predicted trip-duration: 527.65458782

#escooter_model.predict([[5, 1, 1000]]) # 5 PM, 1000 m
#predicted trip-duration: 534.93964914

#escooter_model.predict([[9, 1, 1000]]) # 9 PM, 1000 m
#predicted trip-duration: 542.22471045
