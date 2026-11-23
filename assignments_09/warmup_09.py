# Part 1: Warmup

# --- Supabase Connection ---
# Q1
# In a comment block, answer: what are the two pieces of information supabase-py needs to connect to your project? 
# Where do you find them in the Supabase dashboard, and why should they never be hardcoded in a Python script?
"""
Supbase needs two pieces of information to identify and authenticate a client:
1. Project URL - the unique URL for the project. This is the address the Python script talks to.
2. API key (anon key) - the long string that identifies the application. 

These are found in the project dashboard by clicking the gear icon in the left sidebase, then the API. 

API kets should never be hardcoded in a Python script because it can be scraped within minutes by automated bots. It should be
stored in a .env file. 
"""

# Q2
# Write a function get_client() that:
# Loads your credentials from environment variables using python-dotenv
# Creates and returns a Supabase client
# The function should raise a clear error if either environment variable is missing.
import os
from dotenv import load_dotenv
from supabase import create_client
import datetime

def get_client():
    # read .env and sets environment variables
    load_dotenv()

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    # Raise error if either environment variable is missing.
    if not SUPABASE_URL:
        raise ValueError("Need a Supabase project URL.")

    if not SUPABASE_KEY:
        raise ValueError("Need a Supabase API key.")

    # Create a client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    return supabase

# Q3
# In a comment block, answer: what is Row Level Security (RLS), and why did you disable it on your tables for this 
# course? In what kind of real-world application would you want to keep it enabled?
"""
RLS allows you to define fine-grained access policies. For example, "a user can only read their own rows." 

It was disabled in this project because RLS can block a lot, so the project may get bogged down by debugging any empty results.

There are multiple real-world applications where you would want to keep RLS enabled. For example, any projects that may deal with
personal data, such as health records, you would want RLS enabled. 
"""
# --- supbase-py CRUD ---
# Q1
# Write a function insert_test_record(supabase) that inserts a single row into weather_raw with today's date and plausible values for
# all four weather columns. Run it to confirm it works, then add a comment: what would happen if you ran the function twice? How would
# you change the call to make it safe to run multiple times?
def insert_test_record(supabase):
    records = [
        {
            "date": datetime.date.today().isoformat(),
            "temperature_2m_max": 59,
            "temperature_2m_min": 43,
            "precipitation_sum": 0.4,
            "wind_speed_10m_max": 3,
        }
    ]
    print(f"Prepared {len(records)} records")
    print("First record:", records[0])

    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )

    print(f"Upserted {len(response.data)} rows into weather_raw")

insert_test_record(get_client())

"""
Running the function twice would update the existing row, but it would not duplicate the data.
The call is safer because already because it uses 'upsert' versus 'insert'. An 'insert' would raise
a duplicate-key error on the second run. 
"""

# Q2
# Write a function get_records_by_date_range(supabase, start, end) that returns all rows from weather_raw where date >= start and date 
# <= end. The function should return the list of row dictionaries. Test it with a date range that includes the row you inserted in Q1 
# and print the result.
def get_records_by_date_range(supabase, start, end):

    response = supabase.table("weather_raw").select("*").gte("date", start).lte("date", end).execute()

    return response.data

print(get_records_by_date_range(get_client(), "2026-11-20", "2026-11-23"))

# Q3
# In a comment block, explain the difference between insert and upsert in supabase-py. Give a concrete example of when you would choose 
# each. Then write a function safe_upsert(supabase, records) that upserts a list of records into weather_raw using date as the conflict 
# key and prints the number of rows affected.
"""
Using upsert rather than insert allows a script to run as many times as needed. On the first run, it will insert all 365 rows. On subsequent
runs, it updates them in place. That way, the table always ends up in a consistent state. 

For the weather pipeline, upsert is a preferred method because there is a natural key, date. If you backfill 365 days of historical weather from
an API and the script dies at 200. The bug is fixed and the script re-run. With insert, first row would raise a duplicate-key error since the dates
already exist. With upsert, the existing rows are updated and the remaining are insert. The table would end up with exactly one row per day no matter how
many times the script runs. 

Insert is the preferred method when every row is a new event with no natural key, and a duplicate should raise an error rather than overwrite.
"""

def safe_upsert(supabase, records):
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )

    print(f"Upserted {len(response.data)} rows into weather_raw")

safe_upsert(get_client(), [        {
            "date": "2026-11-21",
            "temperature_2m_max": 58,
            "temperature_2m_min": 42,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 2,
        },
        {
            "date": datetime.date.today().isoformat(),
            "temperature_2m_max": 59,
            "temperature_2m_min": 43,
            "precipitation_sum": 0.4,
            "wind_speed_10m_max": 3,
        },])

# --- Idempotency ---
# Q1
# "Idempotency" means that running an operation multiple times produces the same result as running it once. In a comment block, explain 
# why idempotency matters for a data pipeline. Give one concrete example of what goes wrong in a non-idempotent pipeline when the script 
# crashes halfway through and is restarted.
"""
Rerunning a data pipeline can create duplicate data. Idempotency prevents duplicate data, makes retries safe, handles glitches/timeouts,
keeps a system consistent, and is critical for a data pipeline. If a script crashes halway through and is restarted, a non-idempotent pipeline
may create another entry. This can create duplicative data. 
"""

