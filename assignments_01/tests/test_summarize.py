# Task 5: The Test Suite
import json
import pytest 
from pathlib import Path
from weatherkit import DailySummary, WeatherResponse, HourlyReading, DailyAggregator

DEMO = Path("tests")
DEMO.mkdir(exist_ok=True)

# Path(__file__).parent.parent rather than plain relative path because it finds files related to the script's location, rather than relative to the working directory. 
with open(Path(__file__).parent.parent / "weather_raw.json") as f:
    data = json.load(f)

# Grouping works: a hand-built list spanning two dates produces two summaries
def test_groups():
    reading_1 = HourlyReading("2026-04-10T08:00", 24.0, 2.0)
    reading_2 = HourlyReading("2026-04-10T09:00", 24.5, 2.0)
    reading_3 = HourlyReading("2026-05-10T08:00", 25.0, 1.0)
    reading_4 = HourlyReading("2026-05-10T09:00", 25.5, 1.0)
    
    readings = [reading_1, reading_2, reading_3, reading_4]

    agg = DailyAggregator(min_hours = 2)
    summaries = agg.summarize(readings)

    assert len(summaries) == 2 

# temp_max and temp_min are correct for a known small input.
def test_temp_range():
    day = DailySummary("2026-04-10", 27.0, 24.0, 4.0, 24)
    assert day.temp_range() == 3

# precipitation_sum adds up correctly. Use pytest.approx.
def test_precipitation_sum():
    reading_1 = HourlyReading("2026-04-10T08:00", 24.0, 2.0)
    reading_2 = HourlyReading("2026-04-10T09:00", 24.5, 2.0)
    reading_3 = HourlyReading("2026-05-10T08:00", 25.0, 1.0)
    reading_4 = HourlyReading("2026-05-10T09:00", 25.5, 1.0)
    
    readings = [reading_1, reading_2, reading_3, reading_4]

    agg = DailyAggregator(min_hours = 2)
    summaries = agg.summarize(readings)

    assert summaries[0].precipitation_sum == pytest.approx(4.0)

# A day with fewer than min_hours readings is dropped, and its date appears in incomplete_days().
def test_short_day():
    reading_1 = HourlyReading("2026-04-10T08:00", 24.0, 2.0)
    reading_2 = HourlyReading("2026-04-10T09:00", 24.0, 2.0)
    reading_3 = HourlyReading("2026-04-10T10:00", 24.0, 2.0)

    readings = [reading_1, reading_2, reading_3]

    agg = DailyAggregator(min_hours = 5)
    summaries = agg.summarize(readings)

    incomplete_summaries = agg.incomplete_days()
    
    assert summaries == []
    assert incomplete_summaries == ["2026-04-10"]

# Lowering min_hours causes that same day to be kept -- proving the parameter is actually consulted rather than ignored.
@pytest.mark.parametrize(
    "min_hours, expected_summaries, expected_incomplete",
    [
        (2, 1, []),
        (3, 1, []),
        (4, 0, ["2026-04-10"]),
        (24, 0, ["2026-04-10"])
    ]
)

def test_min_hours(min_hours, expected_summaries, expected_incomplete):
    reading_1 = HourlyReading("2026-04-10T08:00", 24.0, 2.0)
    reading_2 = HourlyReading("2026-04-10T09:00", 24.0, 2.0)
    reading_3 = HourlyReading("2026-04-10T10:00", 24.0, 2.0)

    readings = [reading_1, reading_2, reading_3]

    agg = DailyAggregator(min_hours = min_hours)
    summaries = agg.summarize(readings)

    incomplete_summaries = agg.incomplete_days()

    assert len(summaries) == expected_summaries
    assert incomplete_summaries == expected_incomplete