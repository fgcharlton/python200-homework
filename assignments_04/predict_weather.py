# Task 5: The Script
# Write predict_weather.py that uses the real saved model in models/weather_classifier.pkl
"""Classify a few sample days using the saved weather model."""

from weather_model import WeatherClassifier

SAMPLE_DAYS = [
        {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 12},   # good day
        {"temperature_2m_max": 32, "temperature_2m_min": -1, "precipitation_sum": 5.0, "wind_speed_10m_max": 35},   # bad day
        {"temperature_2m_max": 7, "temperature_2m_min": 0, "precipitation_sum": 2.9, "wind_speed_10m_max": 29.9},    # borderline day
        {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 10.0, "wind_speed_10m_max": 35},  # bad day
]

def main() -> None:
    """Load the component and print a prediction for each sample day."""
    classifier = WeatherClassifier("models/weather_classifier.pkl")
    for day, result in zip(SAMPLE_DAYS, classifier.predict(SAMPLE_DAYS)):
        print(f"{day['temperature_2m_max']:.0f}C, {day['precipitation_sum']:.0f}mm rain, "
              f"{day['wind_speed_10m_max']:.0f} km/h wind -> {result.label} ({result.probability:.2f})")

if __name__ == "__main__":
    main()

# What would happen if you omitted the guard and another script imported predict_weather.py to reuse a helper
# Importing the file sets '__name__' to the module's name instead, so the guarded bloack is skipped. This is what lets one file provided importable functions and run as a script.

# Task 6: Reflection
# Your predict() selects the four features in a fixed order rather than using whatever order the caller passed. Describe a concrete bug this prevents.
# If features are passed in a different order than the model trained on, the model silently produces wrong predictions.
# Selecting them explictly means the caller can pass keys in any order and still get correct results, and a missing feature is caught rather than silently misaligned. 

# The graded tests train their own tiny model in a fixture instead of loading models/weather_classifier.pkl. Why is that a better design for a test suite than depending on the real file?
# Running from a tiny model in a fixture keeps the tests self-contained and fast, and allows each run to start from a clean temporary directory rather than depending on a file on disk. 

# In Week 10 a pipeline will import WeatherClassifier and call predict() on batches of weather records from a database. Name one thing about your component that makes it ready for that, and one thing you might add before it runs in production.
# The pipeline is equipped to handle values as explicitly stated. That is, only as float values (e.g., precipitation_sum = 3.0). However, it is not equiped if the data has values that I don't expect (e.g., precipitation_sum = "lots").