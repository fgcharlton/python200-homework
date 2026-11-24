-- Step 5: Verify with SQL 

-- Row Count and Date Range
-- Write one query that returns the total number of rows in weather_raw, the earliest date, and the latest date.
-- Use COUNT, MIN, and MAX.
SELECT COUNT(*) as total_rows,
MIN("date") as earliest_date,
MAX("date") as latest_date
FROM weather_raw;


-- Missing Values
-- Write one query that counts how many rows in weather_raw have a NULL value in at least one of the four weather 
-- columns. Use IS NULL and OR.
SELECT COUNT(*) as missing_rows
FROM weather_raw
WHERE temperature_2m_max IS NULL
   OR temperature_2m_min IS NULL
   OR precipitation_sum  IS NULL
   OR wind_speed_10m_max IS NULL;

-- Do your SQL results match your Python results from Step 4?
-- Yes, my results match the Python results from step 4. There are 365 rows of daily weather data. The earliest date is 2023-01-01 and the latest date is 2023-12-31.

-- Is the row count exactly what you expected from one year of daily data?
-- Yes, there are 365 rows of daily weather data, which is in alignment with the number of days in a year. 

-- Is the latest date what you expected?
-- Yes, the latest date is 2023-12-31, which is in alignment with the last date in the year. 