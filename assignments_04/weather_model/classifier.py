# Part 2: Project - The weather_model Component

# Task 1: The Prediction Dataclass
# In weather_model/classifier.py, write a dataclass Prediction with two type-hinted fields
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

@dataclass 
class Prediction:
    """The model's verdict for a single day."""
    label: str              # "good" or "skip"
    probability: float      # probability the day is good for running

# Task 2: The WeatherClassifier Class
# Define features
FEATURES = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]

# Write a WeatherClassifier class
class WeatherClassifier:
    """Loads a saved weather pipeline once and predict on many days."""
    def __init__(self, model_path: str | Path):
        """Load the saved pipeline from disk
        
        Args:
            model_path: Path to a .pkl file saved with joblib.

        Raises:
            FileNotFoundError: IF no file exists at model_path
        """
        model_path = Path(model_path)   # Accept a path to a saved .pkl file
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model file at {model_path}. Train and save a model first."
            )   # Raise a FileNotFoundError if no file exists
        self.model = joblib.load(model_path)    # Load the pipeline and store it on self
        self.features = FEATURES 

    # Write predict method
    def predict(self, days: list[dict]) -> list[Prediction]:    # Accepts a list of dictionaries
        """Predict wether each day is good for running.
        
        Args:
            days: A list of dictionaries, each with the four feature keys.

        Returns:
            One prediction per input day, in the same order.

        Raises:
            ValueError: If any required feature is missing.
        """
        if not days:
            return []   # Returns an empty list if days is empty

        frame = pd.DataFrame(days)
        missing = [f for f in self.features if f not in frame.columns]
        if missing:
            raise ValueError(f"missing required features: {missing}")   # Riase a ValueError if missing a required feature

        X = frame[self.features]    # Select features in a fixed model
        labels = self.model.predict(X)  
        probabilities = self.model.predict_proba(X)[:, 1]

        return [
            Prediction("good" if int(label) == 1 else "skip", float(prob))
            for label, prob in zip(labels, probabilities)
        ]   # Return list of Prediction objections, converting 1 to "good" amd 0 to "skip"

