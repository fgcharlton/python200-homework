# Task 4: The Aggregation -- weatherkit/summarize.py
from dataclasses import dataclass
from weatherkit.records import HourlyReading

@dataclass 
class DailySummary:
    """Daily summary of top level weather in Charlotte, NC"""
    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """The day's temperature swing in Celsius (high minus low)"""
        return self.temp_max - self.temp_min

class DailyAggregator:
    def __init__(self, min_hours: int = 24):
        self.min_hours = min_hours 
        self._dropped_dates: list[str] = []

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """Format hourly readings into daily summaries
        
        Args:
        readings: A validated HourlyReading containing the hourly block.

        Returns:
        A list of DailySummary objects, summarized by day, in order.
        """
        by_date: dict[str, list[HourlyReading]] = {}
        for r in readings:
            by_date.setdefault(r.timestamp[:10],[]).append(r)
        
        summaries = []
        for date, day_readings in by_date.items():
            if len(day_readings) < self.min_hours:
                self._dropped_dates.append(date)
                continue
            summaries.append(DailySummary(
                date=date,
                temp_max=max(r.temperature_c for r in day_readings),
                temp_min=min(r.temperature_c for r in day_readings),
                precipitation_sum=sum(r.precipitation_mm for r in day_readings),
                hours_observed=len(day_readings),
            ))
        return sorted(summaries, key=lambda s: s.date)

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """Format incomplete hourly readings

        Returns:
        A list of dropped dates (strings), in the order they were encountered
        """
        self._dropped_dates = []
        self.summarize(readings)
        return self._dropped_dates 