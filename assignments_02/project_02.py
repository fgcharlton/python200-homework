# Part 2: Mini-Project -- Predicting the Daily High

# Task 1: Fetch the data
import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 36.0726,    # Latitude of Greensboro, NC
    "longitude": -79.7915,   # Longitude of Greensboro, NC 
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}

response = requests.get(url, params=params)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

# Chosen City: Greensboro, NC 

# Print shape of dataset
print("Shape of Greensboro Weather Dataset: ", df.shape)

# Print first five rows of dataset
print("First five rows of Greensboro Weather Dataset: \n", df.head())

# Task 2: Look the Features
# Print df.describe() to confirm there are no missing values
print(df.describe())

print(df.isnull().sum())

# There are no missing values. 

# Print the correlation of each numeric column with temperature_2m_max
print(df.corr(numeric_only=True)["temperature_2m_max"].sort_values())

# Which feature relates most strongly to the daily high?
# Temperature_2m_min has a very strong positive correlation with temperature_2m_max (0.92). This matches the intuition that cold nights go with cold days.

# Create at least one plot that helps understand the data.
import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))
plt.scatter(df["temperature_2m_min"], df["temperature_2m_max"], alpha=0.35, s=12)
plt.xlabel("Daily Low Temperature (°C)")
plt.ylabel("Daily High Temperature (°C)")
plt.title("365 Days of Weather in Greensboro, NC. Each dot is one day.")
plt.grid(alpha=0.3)
plt.savefig('assignments_02/outputs/project2_task2.png', dpi=300, bbox_inches='tight')
plt.show()

# The figure displays 365 days of weather in Greensboro, NC. The data shows multiple points that appear to be spread out 
# but generally have the shape of a diagnoal line going from the bottom left hand corner to the top right hand corner of the graph.
# This appears to be consistent that low daily lows don't appear to have very high daily max temperatures and vice versa.

# Task 3: Engineer a Feature
# Add a binary is_summer column: 1 if the month is June, July, or August, and 0 otherwise.
df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)

print(df["is_summer"].value_counts(dropna=False))

# Do you expect is_summer to help predict the daily high, and why?
# Yes, I do expect it to help predict the daily high. Intuitively, summer days are typically warmer than other times of the year.

# Task 4: Baseline Model
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = df["temperature_2m_min"].values.reshape(-1, 1)    # the lows
y = df["temperature_2m_max"].values    # the highs

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()    # Create
model.fit(X_train, y_train)    # Fit
y_pred = model.predict(X_test)    # Predict

# Print slope
print(f"slope = {model.coef_[0]:.2f}")

# Print RMSE
print(f"RMSE = {np.sqrt(mean_squared_error(y_test, y_pred))}")

# Print R² 
print(f"Single Feature: R² = {model.score(X_test, y_test)}")

# Using the RMSE, how far off is a typical prediction?
# With the low temperature as the only predictor, this model's RMSE comes out to around 3.3°C. This means predictions are typically off by a little more than 3.3°C.

# Task 5: Full Model
# Build a model using four features: temperature_2m_min, precipitation_sum, wind_speed_10m_max, and is_summer

feature_cols = ["temperature_2m_min", "precipitation_sum",
                "wind_speed_10m_max", "is_summer"]
X = df[feature_cols].values
y = df["temperature_2m_max"].values

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_full = LinearRegression().fit(X_train_full, y_train_full)     # Create & Fit
y_train_pred = model_full.predict(X_train_full)      # Predict
y_test_pred = model_full.predict(X_test_full)      # Predict

# Print both train R² and test R²
print(f" Train R²: {r2_score(y_train_full, y_train_pred):.3f}")
print(f" Test R²: {r2_score(y_test_full, y_test_pred):.3f}")

# Print RMSE 
print(f"RMSE = {np.sqrt(mean_squared_error(y_test_full, y_test_pred))}")

# How much does adding features help?
# Adding features increased the train and test R² as well as decreased the RMSE, showing an improvement from the single feature model.
# The train R² increased to 0.867 from the single feature R², which was 0.857. This brings R² closer to 1, meaning the model is improved.
# The test R² increased to 0.898 from the single feature R², which was 0.857. This brings R² closer to 1, meaning the model is improved.
# The RMSE decreased to 2.8°C from 3.3°C. This means predictions are typically off by a little more than 2.8°C, which is an improvement from the single feature model.

# Print each feature name with its coefficient:
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:22s}: {coef:+.3f}")

# Interpret the coefficients.
# temperature_2m_min: The model predicts that for every 1°C increase in the minimum temperature,  the predicted max temperature increases by 0.92°C. Meaning, cold nights tend to lead to cold days, and warm nights to warm days. 
# precipitation_sum: The model predicts that for every extra 1mm of rain, the predicted max temperature decreases by 0.15°C.
# wind_speed_10m_max: The model predicts that for every extra 1 m/s of wind, the predicted max temperature decreases by 0.1°C. 
# is_summer: The model predicts that a summer day is about .6°C higher than a non-summer day with the same low, rain, and wind.

# Which features raise the predicted high, and which lower it?
# Temperature_2m_min and is_summer tend to raise the predicted high, while precipitation_sum and wind_speed decrease it. 

# Are any signs surprising?
# No, I believe these findings are consistent with what you'd expect. Meaning, high minimums and summer days lead to higher maximum temperatures. And, rain and wind tend to decrease the maximum daily temperature.

# Compare train R² and test R². Are they close, or is there a gap, what does this tell us?
# The train R² is 0.867 and the test R² is 0.898. The gap is relatively small, indicating the model is a good fit. This means that the model generalizes well.

# Task 6: Evaluate and Summarize
plt.figure(figsize=(10,6))
plt.scatter(y_test_pred, y_test_full, alpha=0.6, edgecolors="k", linewidth=0.5)
plt.plot([y_test_full.min(), y_test_full.max()],
         [y_test_full.min(), y_test_full.max()],
         "r--", label="Perfect fit")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Predicted vs. Actual")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('assignments_02/outputs/predicted_vs_actual_high.png', dpi=300, bbox_inches='tight')
plt.show()

print(X_test_full.shape)
# The size of the dataset and the test set.
# The size of the dataset is 365 rows with 4 columns. 
# The size of the dataset is 73 rows with 4 columns. This is consistent with the 80% for training, 20% for testing. 

# The RMSE and R² of full model, and what a typical error of that size means on a temperature scale.
# The RMSE is 2.8°C, indiciating predictions are typically off by 2.8°C. This is an improvement from the single feature model which had an RSME of 3.3°C. The R² is 0.898, which is an improvement from the single feature model. 

# Which feature has the largest effect on the predicted high, and in which direction.
# The temperature minimum has the biggest impact on predicted high. The model predicts that for every 1°C increase in the minimum temperature,  the predicted max temperature increases by 0.92°C.

# One result that suprised me. 
# I was most surprised by the low coefficients. I thought that all features (minimum temperature, precipitation, wind, and summer day) would have a larger impact on the predicted temperature maximum than they did. 
