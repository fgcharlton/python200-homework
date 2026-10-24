# Task 4: The Test Suite
import pytest
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from weather_model import WeatherClassifier, Prediction, FEATURES

# Build a tiny model inside a fixture and save it to a temporary dictory
@pytest.fixture
def model_path(tmp_path):
    """Train a tiny model on obvious days and save it to a temp file."""
    good = [{"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 12}] * 15
    bad = [{"temperature_2m_max": 32, "temperature_2m_min": -1, "precipitation_sum": 5.0, "wind_speed_10m_max": 35}] * 15
    X = pd.DataFrame(good + bad)[FEATURES]
    y = [1] * 15 + [0] * 15

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X, y)

    path = tmp_path / "weather_classifier.pkl"
    joblib.dump(pipe, path)
    return path

# Load a WeatherClassifier 
@pytest.fixture
def clf(model_path):
    """A WeatherClassifier loaded from the saved tiny model."""
    return WeatherClassifier(model_path)

# Run tests
@pytest.mark.parametrize("day, expected", [
    ({"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 12}, "good"),
    ({"temperature_2m_max": 32, "temperature_2m_min": -1, "precipitation_sum": 5.0, "wind_speed_10m_max": 35}, "skip"),
])

# A clearly good day is labeled "good" and a clearly bad day is labeled "bad"
def test_good_skip_days_predicted_correctly(clf, day, expected):
    result = clf.predict([day])
    assert result[0].label == expected

# When breaking code in classifier.py (deliberately return skip for all days), this test caught the change. 

# A returned proability is between 0 and 1
def test_probability_is_between_zero_and_one(clf):
    result = clf.predict([
        {"temperature_2m_max": 10, "temperature_2m_min": 4, "precipitation_sum": 2.0, "wind_speed_10m_max": 15}
    ])
    assert 0.0 <= result[0].probability <= 1.0

# Passing 3 days returns three Prediction objects in order
def test_one_prediction_per_day_in_order(clf):
    days = [
        {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 12},
        {"temperature_2m_max": 32, "temperature_2m_min": -1, "precipitation_sum": 5.0, "wind_speed_10m_max": 35},
        {"temperature_2m_max": 10, "temperature_2m_min": 4, "precipitation_sum": 2.0, "wind_speed_10m_max": 15}
    ]
    results = clf.predict(days)
    assert len(results) == 3 
    assert all(isinstance(r, Prediction) for r in results)

# Passing an empty list returns an empty list
def test_empty_list_returns_empty(clf):
    assert clf.predict([]) == []

# A day missing a required feature raises ValueError 
def test_missing_feature_raises_clear_error(clf):
    with pytest.raises(ValueError, match="missing required features"):
        clf.predict([{"temperature_2m_max": 10, "temperature_2m_min": 4, "precipitation_sum": 2.0}])    # missing wind_speed_10m_max

# Constructing a WeatherClassifier with a path to a missing file raises FileNotFound 
def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model file"):
        WeatherClassifier(tmp_path / "does_not_exist.pkl")