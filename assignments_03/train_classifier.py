# Task 1: Fetch the DAta
import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 36.044659,
    "longitude": -79.766235,
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

# Task 2: Engineer the Label
# Print df.describe() to confirm there are no missing values
print(df.describe())

print(df.isnull().sum())

# There are no missing values. 

# Create a good_for_running column
df["good_for_running"] = ((7.0 <= df["temperature_2m_max"]) & (df["temperature_2m_max"] <= 28.0) & (df["temperature_2m_min"] >= 0.0) & (df["precipitation_sum"] < 3.0) & (df["wind_speed_10m_max"] < 30.0)).astype(int)

# Print class balance of good days and skip days
print("Balance of Good Days and Skip Days\n", df["good_for_running"].value_counts())

# What fraction of days are good, and does that seem reasonable for Greensboro, NC?
# There were 181 good days for running, while there were 184 skip days. This does seem reasonable for Greensboro, NC as we do have about an even split of nice to not so nice days.
# My data is pretty balanced. 

# Task 3: Build and Evaluate a KNN Classifier 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[features]
y = df["good_for_running"]

# Split the data into training and test sets (80/20, random_state=42, stratify on label)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Use 5-fold cross-validation on the training data to choose k from at least five values
for k in [1, 3, 5, 7, 9, 11, 15, 19, 25]:
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}:   mean={scores.mean():.3f}    std={scores.std():.3f}")

# The best cross-validation score is around k=1. The larger values start to fall off. 

# Build Pipeline of StandardScaler and KNeighborsClassifier
best_k = 1
knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k)),
])

# Fit model
knn_model.fit(X_train, y_train)

# Print accuracy
print("Final test accuracy for KNN Model:", accuracy_score(y_test, knn_model.predict(X_test)))

# Print classification report
print(classification_report(y_test, knn_model.predict(X_test), target_names=["skip","good"]))

# The model does well on both classes. Recall for "good" is high, meaning it catches nearly every good runnning day.
# Precision for "good" is a little lower, meaning it occassionally labels a bad day as good. 

# Task 4: Build and Evaluate a Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression

# Build a Pipeline of StandardScaler and LogisticRegression(max_iter=1000)
logreg_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])

# Fit model
logreg_model.fit(X_train, y_train)

# Print test accuracy, precision, recall, F1, and classification report
print("Final test accuracy for Logistic Regression Model:", accuracy_score(y_test, logreg_model.predict(X_test)))

print(classification_report(y_test, logreg_model.predict(X_test), target_names=["skip","good"]))

# Accuracy is about 0.82. For "skip" days, precision is about 0.90 and recall is 0.73. For "good" days, precision is about 0.77 and recall is 0.92. 

# Print coefficients and interpret them
logreg = logreg_model.named_steps["logreg"]
for name, coef in zip(features, logreg.coef_[0]):
    print(f"{name:20s}: {coef:+.2f}")

# Precipitation has the largest negative coefficient, which means more precipitation pushes the day towards "skip". Minimum temperature, maximum temperature, and wind coefficients are small, meaning they rarely decide the outcome.
# This is consistent with what would be expected because Greensboro, NC has a pretty mild temperatures as well as wind patterns. 

# Which conditions most strongly push a day toward "skip"?
# Precipitation has the strongest push towards a "skip" day due to its large negative coefficient. 

# Task 5: Compare and Choose
# Which model had a high accuracy on the data?
# KNN had the highest accuracy on the data (0.92), while the logistic regression had a lower accuracy (0.82).

# For a running app, is the accuracy the metric you care about most, or would you weigh precision or recall more heavily? Explain in terms of false positives and false negatives.
# For a running app, I would weigh precision and recall more heavily than accuracy. Recall helps decribe how many true positives your model catches, meaning how many days that were labeled as good were truly good. 
# Looking at recall is particularly important for when false negatives are costly. In this case a false negative would mean the app tells you to skip a good day.
# Precision looks at how many good days were really good day. A low precision would mean many false positives, meaning a bad day was labeled as good.
# Neither a false positive or false negative in this case is catastrophic. However, we want our app to be find the true good days so that people either don't miss a good day or go out on a bad day.

# Which model will you deploy, and why? Consider not only the scores, but also model size, prediction speed, interpretability, and whether you want probability confidence scores. 
# Even though the KNN model did better on accuracy, precision, recall, and F1, I plan to deploy the logistic regression model. 
# Logistic regression allows me to produce smooth probabilities that give confidence scores, produce coefficients that help explain the data, and teh saved model is small compared to KNN. 
# The logistic regression model still does relatively wellhaving a high recall on good days (0.92), meaning the model is relatively good at capturing truly good running days. 

# Task 6: Save the Model
import joblib

# Save chosen fitted pipeline
joblib.dump(logreg_model, "models/weather_classifier.pkl")
print("Model saved.")

# Save a metadata file
import json
import sys
import sklearn

metadata = {
    "python_version": sys.version,
    "sklearn_version": sklearn.__version__,
    "city": "Greensboro, NC",
    "features": features,
    "label": "good_for_running",
    "test_accuracy": 0.822,
    "test_f1": 0.81,
    "label_rule": {
        "temperature_2m_max": "7 to 28 C",
        "temperature_2m_min": ">= 0 C",
        "precipitation_sum": "< 3 mm",
        "wind_speed_10m_max": "<30 km/h",
    },
}

with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Metadata saved.")