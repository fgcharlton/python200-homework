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

# Use @pytest.fixture to define readings 
# Create first readings list
@pytest.fixture
def readings():
    """Hand-built list spanning two dates"""
    return [
        HourlyReading("2026-04-10T08:00", 24.0, 2.0),
        HourlyReading("2026-04-10T09:00", 24.5, 2.0),
        HourlyReading("2026-05-10T08:00", 25.0, 1.0),
        HourlyReading("2026-05-10T09:00", 25.5, 1.0)
    ]

# Create shortened readings list
@pytest.fixture
def short_day_readings():
    """Hand-built list spanning one short day"""
    return [
        HourlyReading("2026-04-10T08:00", 24.0, 2.0),
        HourlyReading("2026-04-10T09:00", 24.0, 2.0),
        HourlyReading("2026-04-10T10:00", 24.0, 2.0)
    ]

# Grouping works: a hand-built list spanning two dates produces two summaries
def test_groups(readings):
    agg = DailyAggregator(min_hours = 2)
    summaries = agg.summarize(readings)
    assert len(summaries) == 2 

# temp_max and temp_min are correct for a known small input.
def test_temp_range():
    day = DailySummary("2026-04-10", 27.0, 24.0, 4.0, 24)
    assert day.temp_range() == 3
# Broke temp_range() function received the following error
#====================================== 8 passed in 0.12s =======================================
#(base) Fishers-Air:assignments_01 fishercharlton$ pytest tests/test_summarize.py -v
#===================================== test session starts ======================================
#platform darwin -- Python 3.13.5, pytest-9.1.1, pluggy-1.5.0 -- /opt/anaconda3/bin/python
#cachedir: .pytest_cache
#rootdir: /Users/fishercharlton/CTD-repos/python200-homework/assignments_01
#plugins: anyio-4.7.0
#collected 8 items                                                                              

#tests/test_summarize.py::test_groups PASSED                                              [ 12%]
#tests/test_summarize.py::test_temp_range FAILED                                          [ 25%]
#tests/test_summarize.py::test_precipitation_sum PASSED                                   [ 37%]
#tests/test_summarize.py::test_short_day PASSED                                           [ 50%]
#tests/test_summarize.py::test_min_hours[2-1-expected_incomplete0] PASSED                 [ 62%]
#tests/test_summarize.py::test_min_hours[3-1-expected_incomplete1] PASSED                 [ 75%]
#tests/test_summarize.py::test_min_hours[4-0-expected_incomplete2] PASSED                 [ 87%]
#tests/test_summarize.py::test_min_hours[24-0-expected_incomplete3] PASSED                [100%]

#=========================================== FAILURES ===========================================
#_______________________________________ test_temp_range ________________________________________

#    def test_temp_range():
#        day = DailySummary("2026-04-10", 27.0, 24.0, 4.0, 24)
#>       assert day.temp_range() == 3
#E       AssertionError: assert 51.0 == 3
#E        +  where 51.0 = temp_range()
#E        +    where temp_range = DailySummary(date='2026-04-10', temp_max=27.0, temp_min=24.0, precipitation_sum=4.0, hours_observed=24).temp_range

#tests/test_summarize.py:31: AssertionError
#=================================== short test summary info ====================================
#FAILED tests/test_summarize.py::test_temp_range - AssertionError: assert 51.0 == 3
#================================= 1 failed, 7 passed in 0.13s ==================================

# precipitation_sum adds up correctly. Use pytest.approx.
def test_precipitation_sum(readings):
    agg = DailyAggregator(min_hours = 2)
    summaries = agg.summarize(readings)
    assert summaries[0].precipitation_sum == pytest.approx(4.0)

# A day with fewer than min_hours readings is dropped, and its date appears in incomplete_days().
def test_short_day(short_day_readings):
    agg = DailyAggregator(min_hours = 5)
    summaries = agg.summarize(short_day_readings)
    incomplete_summaries = agg.incomplete_days(short_day_readings)
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

def test_min_hours(short_day_readings, min_hours, expected_summaries, expected_incomplete):
    agg = DailyAggregator(min_hours = min_hours)
    summaries = agg.summarize(short_day_readings)
    incomplete_summaries = agg.incomplete_days(short_day_readings)
    assert len(summaries) == expected_summaries
    assert incomplete_summaries == expected_incomplete