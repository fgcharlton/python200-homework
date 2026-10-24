# Part 1: Warmup Exercises

# Q1: Write a ModelInfo that reads a JSON file once in __init__ and stores its contents
from pathlib import Path
import json

# Create JSON file
data = {
    "features": [
        "temperature_2m_max", 
        "temperature_2m_min", 
        "precipitation_sum", 
        "wind_speed_10m_max"
    ]
    }

Path("model_info.json").write_text(json.dumps(data))

class ModelInfo:
    """Load a saved weather pipeline once and predicts on many days."""

    def __init__(self, json_path: str | Path):
        """Load the saved JSON file from disk.
        
        Args:
            json_path: Path to a .json file saved.

        Raises:
            FileNotFoundError: If no file exists at json_path. 
        
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(
                f"No json file at {json_path}. Save a JSON metadata file first."
            )
        with open(json_path) as f:
            self.data = json.load(f)

    def feature_count(self) -> int:
        return len(self.data["features"])

# print feature count
print(ModelInfo("model_info.json").feature_count())

# Why is loading the file in __init__ better than loading it in feature_count() if the method is called many times?
# Loading it in __init__ reads the file once. If it is loaded in feature_count() then it would get called many times. 
# Loading it multiple times can take a long time, so it loading it once in __init__ and keeping it in self.data means each call reads from memory. This saves time.

# Q2: Write a dataclass DayForecast with three type-hinted fields
from dataclasses import dataclass 

@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool


    def summary(self) -> str:
        return f"{self.date}: high {self.high_c}C, {'will rain' if self.will_rain else 'will not rain'}"

day1 = DayForecast(date="2026-10-22", high_c=21.0, will_rain=False)
day2 = DayForecast(date="2026-10-23", high_c=19.0, will_rain=True)

print(day1.summary())
print(day2.summary())

# Name one thing the @dataclass decorator wrote for you that you did not have to write by hand
# A dataclass allows you to avoid writing repetitive boilerplate code for attributes, constructors like __init__, and methods like __repr__.

# Q3: Write a function parse_temperature(text: str) -> float that converts a string like "24.1" to a float
import pytest 

def parse_temperature(text: str) -> float:
    """Convert text to int. Raises ValueError if 'text' cannot be converted to a float."""
    try:
        return float(text)
    except ValueError:
        raise ValueError("temperature")

# Write a pytest
# Pytest that asserts parse_temperature("24.1") returns 24.1
def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)

# Pytest uses pytest.raises(ValueError, match="temperature") to confirm that parse_temperature("hot") raise the right error
def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")

# What would pytest.raises(ValueError) without match fail to catch?
# Match checks the error message, so the test passes only when the specific expected error is raised. 
# Without it, an unrelated 'ValueError' from a typo in the test setup would still make the test pass, giving false confidence. 
