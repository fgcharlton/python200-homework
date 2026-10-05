# Task 5: The Test Suite
import json
from pathlib import Path
from weatherkit import WeatherResponse, HourlyReading, to_readings

# Path(__file__).parent.parent rather than plain relative path because it finds files related to the script's location, rather than relative to the working directory. 
with open(Path(__file__).parent.parent / "weather_raw.json") as f:
    data = json.load(f)

# Test to_readings returns one reading per hour, in order
def test_to_readings():
    weather = WeatherResponse.model_validate(data)
    readings = to_readings(weather)
    assert len(readings) == 168
    assert readings[0].timestamp == weather.hourly.time[0]
    assert readings[-1].timestamp == weather.hourly.time[-1]

# The values in reading i match index i of each input list.
def test_indexes():
    weather = WeatherResponse.model_validate(data)
    readings = to_readings(weather)
    i = 10 
    assert weather.hourly.time[i] == readings[i].timestamp
    assert weather.hourly.temperature_2m[i] == readings[i].temperature_c
    assert weather.hourly.precipitation[i] == readings[i].precipitation_mm

# Two HourlyReading objects with identical fields compare equal.
def test_hourly_reading_identical():
    reading_a = HourlyReading("2023-04-10", 24.0, 2.0)
    reading_b = HourlyReading("2023-04-10", 24.0, 2.0)
    assert reading_a == reading_b 
