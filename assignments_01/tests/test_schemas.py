# Task 5: The Test Suite
import json
import pytest 
from pathlib import Path
from weatherkit import WeatherResponse, HourlyBlock
from pydantic import ValidationError

DEMO = Path("tests")
DEMO.mkdir(exist_ok=True)

# Path(__file__).parent.parent rather than plain relative path because it finds files related to the script's location, rather than relative to the working directory. 
with open(Path(__file__).parent.parent / "weather_raw.json") as f:
    data = json.load(f)

# A valid response validates, and hourly.time has 168 entries
def test_weather_response():
    weather = WeatherResponse.model_validate(data)
    assert len(weather.hourly.time) == 168

# A latitude of 200.0 raises a Validation Error
def test_latitude():
    bad_data = data.copy()
    bad_data["latitude"] = 200.0
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(bad_data)

# Mismatched list lengths raise Validation error
def test_mismatch():
    bad_data = {
        "time": ["2026-02-10","2026-03-10","2026-02-10"],
        "temperature_2m": [24.0, 25.0, 24.0],
        "precipitation": [4.0, 2.0]
    }
    with pytest.raises(ValidationError):
            HourlyBlock.model_validate(bad_data)

# A null inside temperature_2m raises ValidationError.
def test_null():
    bad_data = {
        "time": ["2026-02-10","2026-03-10","2026-02-10"],
        "temperature_2m": [24.0, None, 24.0],
        "precipitation": [4.0, 2.0]
    }
    with pytest.raises(ValidationError):
        HourlyBlock.model_validate(bad_data)