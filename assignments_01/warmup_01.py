## Part 1: Warmup Exercises 

# --- Classess ---
# Question 1: Write a class Therometer that stores a list of temperature readings in Celsius
class Thermometer:
    def __init__(self, location: str, readings: list = None):
        # Assign to self object
        self.location = location
        self.readings = readings if readings is not None else []

    # Actions to execute
    def add(self, reading):
        """Add temperature (in Celsius) to the list"""
        self.readings.append(reading)

    def average(self):
        """Get the average temperature (in Celsius) for chosen location"""
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self):
        """Get the hottest temperature (in Celsius) for chosen location"""
        if not self.readings:
            return None
        return max(self.readings)

    # Question 2: Add a __repr__ to Thermometer
    def __repr__(self):
        return f"{self.__class__.__name__} ({self.location}, n_readings = {len(self.readings)}, average = {sum(self.readings) / len(self.readings)})"

my_thermometer = Thermometer("Greensboro")
my_thermometer.add(31)
my_thermometer.add(23)
my_thermometer.add(17)
my_thermometer.add(24)

# Create a second Thermometer 
second_thermometer = Thermometer("Charlotte")
second_thermometer.add(30)
second_thermometer.add(22)
second_thermometer.add(16)
second_thermometer.add(23)

# Question 1 Responses
print(my_thermometer.average())
print(my_thermometer.hottest())

# Why does average() need to handle the empty case? What would happen without that check?
# Without handling the empty case, you cannot average a list of empty records. It would raise a ValueError.

# Question 2 Responses 
print(my_thermometer)

# Print list of two
print([my_thermometer, second_thermometer])

# Explain what Python displays when a class has no __repr__, and why that is unhelpful when debugging.
# Python shows the type and memory address of an object, which tells you nothing useful for debugging.
# __repr__ returns an understandable string that desribes a Python object, which helps developers know what they are looking at when they are attempting to debug.

# Question 3: Write a second class TemperatureAlert that holds a threshold (a float, defaulting to 30.0) and has one method:
class TemperatureAlert:
    def __init__(self, threshold: float = 30.0):
        # Assign to self object
        self.threshold = threshold
        
    # Actions to execute
    def breaches(self, thermometer):
        """Create a threshold to check temperatures"""
        return [reading for reading in thermometer.readings if reading > self.threshold]

# Create two alerts with different thresholds
alert1 = TemperatureAlert()     # Default is 30.0
alert2 = TemperatureAlert(20.0)

# Question 3 Responses
print(alert1.breaches(my_thermometer))
print(alert2.breaches(my_thermometer))

# Why is the threshold stored on Temperature Alert rather than passed as an argument to breaches()?
# The threshold is a property of Temperature Alert. Storing it means that you can create many alerts without repeating the threhold argument. 

# What advantage does that give you if you have twenty thermometers to check?
# There is no need to pass the threshold each time. In addition, each alert can be reused.


# --- Dataclasses, Type Hints, and Docustrings ---

# Question 1: Rewrite class as a dataclass, add type hints to every field and a docstring describing what it represents
from dataclasses import dataclass, FrozenInstanceError, field

# Question 2: Make Station frozen
@dataclass(frozen = True)
class Station: 
    """Defining station by defining station_id, station name, latitude, longitude, and elevation (in meters)"""
    station_id: int
    name: str
    latitude: float
    longitude: float
    elevation: float 

station_a = Station(1, "Station A", 40.741895, -73.989308, 12)
station_b = Station(1, "Station A", 40.741895, -73.989308, 12)

# Question 1 Responses
print(station_a == station_b)

# Why did I get this result? 
# Dataclass objects can be compared direcly using ==. Since the objects are the same it results in 'True'.

# What would it have been with the original class?
# The original class would results in 'False' because there is no __eq__() method in the class.

# Question 2 Responses
try:
    station_b.name = "Station B"
except FrozenInstanceError as e:
    print("Do not let the script crash:", e)

stations = {
    Station(1, "Station A", 40.741895, -73.989308, 12),
    Station(1, "Station A", 40.741895, -73.989308, 12), # same values -- deduplicated
    Station(2, "Station C", 40.79826, -73.945639, 6)
}

print(len(stations))

# What does frozen = True give you besides immutability, and why is that useful here?
# A frozen dataclass also becomes hashable, so you can use it as a dictionary key or put it in a set. This enable value-based deduplication. 

# First try writing the default as stations: list[Station] = []
#@dataclass
#class StationBatch:
#    region: str
#    stations: list[Station] = []

# Paste error I get
# ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory

# Write a dataclass StationBatch
@dataclass
class StationBatch:
    region: str
    stations: list[Station] = field(default_factory = list)

    def add(self, station: Station) -> None:
        """Add station to list of stations"""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Get station with highest elevation"""
        if not self.stations:
            return None
        return max(self.stations, key=lambda station: station.elevation)

# Question 3 Responses 
# Explain in that comment why Python refuses the first version.
# A default value is created once when the class is defined, so every instance would share the same list. 
# Appending to one batch would append to all of them. Dataclasses detects the problem and refuses to define the class.

# --- Pydantic ---

# Question 1: Write a Pydantic model Reading with these fields
from pydantic import BaseModel, Field, ValidationError, model_validator

class Reading(BaseModel):
    station_id: str = Field(min_length = 3, description = "Station ID")
    timestamp: str = Field(min_length = 1)
    temperature_c: float = Field(ge = -90, le = 60, description = "Temperature in Celsius")
    humidity: float = Field(ge = 0, le = 100, description = "Humidity")

    # Question 4: Add a model_validator(mode="after") to Reading
    @model_validator(mode="after")
    def failed_sensor(self):
        """Reject any reading where humidity = 0.0 and temperature_c < -40, indicates failed sensor"""
        if self.humidity == 0.0 and self.temperature_c < -40.0:
            raise ValueError(
                f"Humidity is {self.humidity} and Temperature is {self.temperature_c} indicating a failed sensor."
            )
        return self 

# Question 1 Responses
print(Reading(station_id = "101", timestamp = "2026-10-03", temperature_c = 27.0, humidity = 94.0))

# Question 2: Show three seperate 
# A missing required field
try:
    Reading(station_id = "101", temperature_c = 27.0, humidity = 94.0)
except ValidationError as e:
    print(f"{e.error_count()} problems found\n")
    for err in e.errors():
        print(f"  field={err['loc']}  type={err['type']}  msg={err['msg']}")

# A temperature_c of 150.0 
try:
    Reading(station_id = "101", timestamp = "2026-10-03", temperature_c = 150.0, humidity = 94.0)
except ValidationError as e:
    print("\n", e)

# A humidity of "very humid"
try:
    Reading(station_id = "101", timestamp = "2026-10-03", temperature_c = 27.0, humidity = "very humid")
except ValidationError as e:
    print(e)

# Construct a new Reading where temperature_c is passed as a string and humidity is passed as the integer
try:
    r = Reading(station_id = "101", timestamp = "2026-10-03", temperature_c = "21.5", humidity = 40.0)
    print(f"temperature = {r.temperature_c}, humidity = {r.humidity}")
except ValidationError as e:
    print(f"{e.error_count()} problems found\n")
    for err in e.errors():
        print(f"  field={err['loc']}  type={err['type']}  msg={err['msg']}")

# Why does Pydantic accept "21.5" but reject "very humid"? State the rule in your own words.
# Pydantic coerce a value where there is only one reasonable interpretation of it. So, "21.5" can be interpreted as 21.5, but "very humid" could be any high humidity value.

# Question 3: Trigger several errors at once
try:
    Reading(station_id = "1", temperature_c = "very hot", humidity = 94.0)
except ValidationError as e:
    print(f"{e.error_count()} problems found\n")
    for err in e.errors():
        print(f"  field={err['loc']}  type={err['type']}  msg={err['msg']}")

# Question 3 Responses
# How many errors were reported, and why is reporting all of them at once more useful than stopping at the first?
# 3 errors were reported. Receiving the entire list of what is wrong at once allows you to fix the problems together instead of one at a time, saving a great deal of time and effort. 

# Question 4 Responses
try:
    print(Reading(station_id = "101", timestamp = "2026-10-03", temperature_c = -42.0, humidity = 0.0))
except ValidationError as e:
    print(e)

# Explain why this rule cannot be expressed with Field constraints alone.
# A model_validator runs after all fields are populated, so it can compare them. 
# Neither value raises an error on its own, but together they do. A field check would not detect both together.

# --- pytest --- 

# Question 1: Write a function celsius_to_fahrenheit
import pytest 

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert temperature from celsius to fahrenheit"""
    return (celsius * 9/5) + 32

def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

# Explaining why pytest.approx was necessary.
# pytest.approx compares float values with some relative tolerance. 
# Math with float numbers can sometimes produce round errors, so pytest.approx allows for passing if values are close enough.

# Question 2: mean(values: list[float]) -> float
def mean(values: list[float]) -> float:
    """Average values. Raises ValueError if 'values' is empty."""
    if not values:
        raise ValueError("Cannot average an empty list of values.")
    return sum(values) / len(values)

def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])

# What would pytest.raises(ValueError) alone fail to catch that match= catches?
# The match= checks the error message against a regular expression.
# Without it, pytest.raises(ValueError) passes if any ValueError occurs, including those caused by a type in the test setup.
# Essentially it produces a test that passes for the wrong reasons. 

# Question 3: test_mean_values() using @pytest.mark.parametrize 
@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, 2, 3], 2),
        ([5], 5),
        ([-4, -6, -2], -4),
        ([7, 8, 9], 8)
    ]
)
def test_mean_values(values, expected):
    assert mean(values) == expected

# Why is one parametrized test with four cases better than four nearly identical test functions?
# @pytest.mark.parametrize combines identical code into one test that runs several times, reducing code length and improving readability.
# Each test still would get its own line, allowing for its own pass or fail result, so a single broken case does not hide. 

# Question 4: Deliberately break celsius_to_fahrenheit 

# Failure output
# ================================================= test session starts =================================================
# platform darwin -- Python 3.13.5, pytest-9.1.1, pluggy-1.5.0 -- /opt/anaconda3/bin/python
# cachedir: .pytest_cache
# rootdir: /Users/fishercharlton/CTD-repos/python200-homework/assignments_01
# plugins: anyio-4.7.0
# collected 5 items                                                                                                     

# warmup_01.py::test_celsius_to_fahrenheit FAILED                                                                 [ 20%]
# warmup_01.py::test_mean_of_empty_raises PASSED                                                                  [ 40%]
# warmup_01.py::test_mean_values[values0-2] PASSED                                                                [ 60%]
# warmup_01.py::test_mean_values[values1-5] PASSED                                                                [ 80%]
# warmup_01.py::test_mean_values[values2-8] PASSED                                                                [100%]

# ====================================================== FAILURES =======================================================
# _____________________________________________ test_celsius_to_fahrenheit ______________________________________________

#     def test_celsius_to_fahrenheit():
#         assert celsius_to_fahrenheit(0) == 32
# >       assert celsius_to_fahrenheit(100) == 212
# E       assert 257.0 == 212
# E        +  where 257.0 = celsius_to_fahrenheit(100)

# warmup_01.py:242: AssertionError
# =============================================== short test summary info ===============================================
# FAILED warmup_01.py::test_celsius_to_fahrenheit - assert 257.0 == 212
# ============================================= 1 failed, 4 passed in 0.14s =============================================

# What specific values did pytest show you in the failure report, and why is that more useful than a bare "assertion failed"?
# The test showed me the exact values that my test found versus the expected value I gave it. 
# This was more helpful than the bare "assertion failed" because it would allow me to debug more easily, know exactly what failed, why it failed, and what I expected it to be.

