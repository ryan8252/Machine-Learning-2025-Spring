import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Load the data
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

# Select specific important features
selected_features = [
    'cli_day1',
    'cli_day2',
    'cli_day3',
    'ili_day3',
    'wworried_catch_covid_day1',
    'tested_positive_day1',
    'wshop_indoors_day3',
    'wlarge_event_indoors_day3',
    'hh_cmnty_cli_day3',
    'tested_positive_day2',
    'wearing_mask_7d_day1',
    'wearing_mask_7d_day2',
    'hh_cmnty_cli_day2',
    'hh_cmnty_cli_day1',

]



X_train = train_df[selected_features]
y_train = train_df['tested_positive_day3']
X_test = test_df[selected_features]

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Split the data into training and validation sets
X_train_val, X_val, y_train_val, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()
# Train the model
model.fit(X_train_val, y_train_val)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate the model using MSE
mse = mean_squared_error(y_val, y_pred_val)
print(f'MSE: {mse}')

# Make predictions on the test set
y_pred_test = model.predict(X_test_scaled)

# Save the predictions to a submission file
submission_df = pd.DataFrame({'id': test_df['id'], 'tested_positive_day3': y_pred_test})
submission_df.to_csv('./submission_911.csv', index=False)
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler

# # Load the data
# train_df = pd.read_csv('./train.csv')
# test_df = pd.read_csv('./test.csv')

# # 先確認欄位名稱
# print("Train columns:", train_df.columns.tolist())
# print("Test columns:", test_df.columns.tolist())
