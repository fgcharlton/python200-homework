# Task 2: The Boundary -- weatherkit/schemas.py
from pydantic import BaseModel, model_validator, Field

class HourlyBlock(BaseModel):
    """The columnar arrays returned under HourlyBlock"""
    time: list[str] = Field(default_factory = list)
    temperature_2m: list[float] = Field(default_factory = list)
    precipitation: list[float] = Field(default_factory = list)

    @model_validator(mode="after")
    def check_entries(self):
        """The three lists (time, temperature, and precipitation) must be the same length"""
        if not len(self.time) == len(self.temperature_2m) == len(self.precipitation):
            raise ValueError(
                f"Length mismatch: time={len(self.time)}, temperature_2m={len(self.temperature_2m)}, precipitation={len(self.precipitation)}"
            )
        return self 

class WeatherResponse(BaseModel):
    """A daily top level weather response from the Open-Meteo archive for Charlotte, NC"""
    latitude: float = Field(ge=-90, le=90, description="Latitude")
    longitude: float = Field(ge=-180, le=180, description="Longitude")
    timezone: str
    elevation: float
    hourly: HourlyBlock