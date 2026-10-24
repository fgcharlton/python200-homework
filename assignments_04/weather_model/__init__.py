# Task 3: The Package Entry Point
# Write a weather_model/__init__.py so from weather_model import Weather Classifier, Predict works. Include an __all__ list

"""weather_model: a reusable weather classifier component."""

from weather_model.classifier import WeatherClassifier, Prediction, FEATURES

__all__ = ["WeatherClassifier", "Prediction", "FEATURES"]