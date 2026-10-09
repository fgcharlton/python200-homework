from dataclasses import dataclass 
from weatherkit.schemas import WeatherResponse

# Task 3: Inside the Boundary -- weatherkit/records.py
@dataclass
class HourlyReading:
    """Record types for hourly weather observations in Charlotte, NC

    Attributes:
        timestamp: Observation date is a full ISO string, e.g., ""2026-04-13T11:00"
        temperature_c: Hourly temperature in degress Celsius.
        precipitation_mm: Total precipitation in millimeters. 
    """
    timestamp: str
    temperature_c: float
    precipitation_mm: float

def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert columnar hourly block into one reading per hour
    
    Args:
        response: A validated WeatherResponse containing the hourly block.

    Returns:
        A list of HourlyReading objects, one per hour, in order.
    """
    readings = []
    for t, temp, precip in zip(response.hourly.time, response.hourly.temperature_2m, response.hourly.precipitation):
        readings.append(HourlyReading(timestamp=t, temperature_c=temp, precipitation_mm=precip))
    return readings 
    
# Why is HourlyReading a dataclass rather than a Pydantic model, when WeatherResponse is a Pydantic model?
# WeatherResponse is a Pydantic model because it describes outside data that must be checked.
# HourlyReading is a dataclass because it describes inside data my own code just produced. Validating twice would be wasted work. 