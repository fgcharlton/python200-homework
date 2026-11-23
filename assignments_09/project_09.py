# Part 2: Project - Extract + Load Pipeline
import os
import requests 
from dotenv import load_dotenv
from supabase import create_client

# VIDEO LINK for presentation for Extract + Load Pipeline
# https://youtu.be/tmPMr_I2FgU

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Step 1: Extract
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 36.0726,
    "longitude": 79.7915,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}

response = requests.get(url, params=params)

# Catch errors 
response.raise_for_status()
data = response.json()

# Print a summary of the response
daily = response.json()["daily"]
print(f"Fetched {len(daily['time'])} daily records from Open-Meteo")

# Step 2: Transform
records = [
    {
        "date":               daily["time"][i],
        "temperature_2m_max": daily["temperature_2m_max"][i],
        "temperature_2m_min": daily["temperature_2m_min"][i],
        "precipitation_sum":  daily["precipitation_sum"][i],
        "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
    }
    for i in range(len(daily["time"]))
]

# Print first and last record to confirm the transformation looks correct
print("First record:", records[0])
print("Last record:", records[364])

# How many records do you expect for a full year, and how many did you get?
# I expected there to be 365 records because there are 365 days in a year. There were 365 daily records fetched. 

# If numbers differ, what might explain the discrepancy?
# The numbers do not differ. 

# Step 3: Load
db_response = (
    supabase.table("weather_raw")
    .upsert(records, on_conflict="date")
    .execute()
)

print(f"Upserted {len(db_response.data)} rows into weather_raw")

# Confirm the row count in weather_raw does not change. What does this tell you about idempotency?
# Idempotency ensures that the pipeline doesn't add duplicative data into the table. 

# Step 4: Verify
# Print total number of rows in weather_raw
count_response = supabase.table("weather_raw").select("date", count="exact").execute()
print(f"Total rows in weather_raw: {count_response.count}")

# Print earliest and latest dates in the table
first = supabase.table("weather_raw").select("*").eq("date", "2023-01-01").execute()
last = supabase.table("weather_raw").select("*").eq("date", "2023-12-31").execute()

print("First record:", first.data)
print("Last record:", last.data)

# Print the row for 2023-07-04
specific_record = supabase.table("weather_raw").select("*").eq("date", "2023-07-04").execute()

print("2023-07-04 Record:", specific_record.data)