# Task 6: The Script -- report.py
import json
from weatherkit import WeatherResponse, to_readings, DailyAggregator

def main() -> None:
    """Load, validate, aggregate, and print the weather report."""

    # Task 2: The Boundary -- weatherkit/schemas.py test
    # Load data 
    with open("weather_raw.json") as f:
        data = json.load(f)

    weather = WeatherResponse.model_validate(data)

    print(f"Latitude: {weather.latitude}")
    print(f"Timezone: {weather.timezone}")
    print(f"Hourly Observations: {len(weather.hourly.time)}")

    # Task 4: The Aggregation -- weatherkit/summarize.py test
    # Convert readings
    readings = to_readings(weather) 

    # Pull in aggregator
    agg = DailyAggregator()
    # Store observations
    summaries = agg.summarize(readings)

    # Print length and first two observations
    print(len(summaries))
    print("First observation:\n", summaries[0])
    print("Second observation:\n", summaries[1])

    # print the weather table
    print("Date         High     Low     Precipitation    Range")
    print("-" * 55)

    for summary in summaries:
        print(
            f"{summary.date}  "
            f"{summary.temp_max:6.1f}C  "
            f"{summary.temp_min:6.1f}C  "
            f"{summary.precipitation_sum:10.1f}mm  "
            f"{summary.temp_range():6.1f}C"
        )

    # print a warning for any incomplete days
    incomplete = agg.incomplete_days(readings)
    if incomplete:
        print(f"WARNING: Incomplete days dropped: {', '.join(incomplete)}")

if __name__ == "__main__":
    main()

# Explain what would happen if you omitted the guard and someone imported report.py to reuse one of its helper functions.
# The guard directs it to run only when directed, not when imported. If the guard is omitted, it may accidentally run code when importing.

# Task 7: Reflection
# Your WeatherResponse rejects the whole file if a single temperature is null. Is that the right behavior for a weather pipeline? Describe one situation where you would want it, and one where you would rather tolerate the gap. What would you change in the schema to tolerate it?
# It is dependent on the weather pipeline and what you hope to capture. For example, in this notebok, we do a variety of things with temperature, such as calculating max, min, and range of temperatures. In this case, it would be beneficial to reject if we are missing temperature. In situation where we don't necessarily care about temperature variables, we can choose to tolerate missing temperature. It could be helpful to push the file through if you were more interested in precipitation measures. You could make it tolerate a gap if temperature_2m was allowed to contain None.  

# DailyAggregator.min_hours defaults to 24. What goes wrong if a pipeline runs at noon and the day is only half over? How does incomplete_days() help?
# This would cause those dates to be dropped from summaries. However, incomplete_days() still allows those dates to be captured in case additional investigation and debugging measures needs to be taken. 

# You wrote weatherkit as a package rather than one file. Name one concrete thing that becomes easier in Week 10, when a pipeline needs to import this code.
# There are two kinds of Python files, a library module (defines things) and scripts (perform actions). Problems can arise when a singles file tries to be both. In addition, and most importantly, the package can be reused by several scripts without copying that code into the pipeline. This is exactly what happens in Week 10, when a pipeline that is written in Week 4 imports the model component. 