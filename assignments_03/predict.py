# Task 7: Load and Predict
import joblib
import json
import pandas as pd

# Load the pipeline
clf = joblib.load("models/weather_classifier.pkl")

# Load the metadata 
with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)

# Build small DataFrame of at least 4 hypothetical days: a clearly good day, a clearly bad day, and at least one borderline day
new_days = pd.DataFrame(
    [
        {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 12},   # good day
        {"temperature_2m_max": 32, "temperature_2m_min": -1, "precipitation_sum": 5.0, "wind_speed_10m_max": 35},   # bad day
        {"temperature_2m_max": 7, "temperature_2m_min": 0, "precipitation_sum": 2.9, "wind_speed_10m_max": 29.9},    # borderline day
        {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 10.0, "wind_speed_10m_max": 35},  # bad day
    ]
)[metadata["features"]]

# Print the four inputs
print("New Weather Days\n", new_days)

# Print the predicted label (good or skip) and model's confidence
preds = clf.predict(new_days)
probs = clf.predict_proba(new_days)[:,1]

for i, (pred, prob) in enumerate(zip(preds, probs)):
    label = "good for running" if pred == 1 else "skip"
    print(f"Day {i + 1}: {label}  (confidence: {prob:.2f})")

# Task 8: Reflect
# Pick you borderline day. What probability did the model give it? Would you call the model confident or uncertain, and how would you handle a day where it predicts 0.52?

# predict.py has no training code. What would break if someone ran predict.py before train_classifier.py had ever been run?
# How could you make this error message more helpful?

# In Week 10, this model will run inside a cloud pipeline that classifies each day's weather automatically. Name one thing you would change about the current predict.py that would need to change to support that.
