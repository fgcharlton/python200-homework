# Task 7: Load and Predict
import joblib
import json
import pandas as pd

# Load the pipeline
clf = joblib.load("models/weather_classifier.pkl")

# Load the metadata 
with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)
    print("City: ", metadata["city"])
    print("Features: ", metadata["features"])
    print("Accuracy: ", metadata["test_accuracy"])

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
"""My borderline day was labeled as "good" with a 0.79 confidence. So, even though the day is close to the threshold, the model was pretty confident that it was still a good day for running. 
A day that is 0.52 is about the same as flipping a coin."""

# predict.py has no training code. What would break if someone ran predict.py before train_classifier.py had ever been run?
# How could you make this error message more helpful?
"""If train_classifier.py was never ran, the file would not exist so it would not be able to load in the model. This is a loud, immediate failure that tells you what is wrong.
If just the model is saved and not the whole pipeline, when you call .predict() on raw, unscaled data the predictions will be silently wrong with no error.
When you load the data, you can wrap joblib.load in a try/except with a clear message that states 'model file not found; run train_classifier.py first' 
"""
# In Week 10, this model will run inside a cloud pipeline that classifies each day's weather automatically. Name one thing you would change about the current predict.py that would need to change to support that.
"""The current script has no error handling around the model load. In a cloud pipline, a missing pkl file shouldn't crash the script - it should fail with a clear message. """
