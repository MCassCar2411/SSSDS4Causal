import os
from datetime import datetime

import pandas as pd
import requests
from pvlib.location import Location


class WeatherData:
    POSTCODES_API_URL = "https://api.postcodes.io/postcodes/"
    OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

    @staticmethod
    def get_coordinates_from_postcode(postcode):
        try:
            response = requests.get(f"{WeatherData.POSTCODES_API_URL}{postcode}")
            response.raise_for_status()
            data = response.json()
            if data["status"] == 200:
                return data["result"]["latitude"], data["result"]["longitude"]
            else:
                return None, None
        except requests.RequestException:
            return None, None

    @staticmethod
    def get_weather_data(lat, lon, start_date, end_date):
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": [
                "temperature_2m",
                "windspeed_10m",
                "relative_humidity_2m",
                "rain",
                "shortwave_radiation",
            ],
            "timezone": "Europe/London",
        }

        try:
            response = requests.get(WeatherData.OPEN_METEO_URL, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            return None

    @staticmethod
    def calculate_clear_sky_irradiance(lat, lon, times):
        location = Location(latitude=lat, longitude=lon, tz="Europe/London")
        clear_sky = location.get_clearsky(times)
        return clear_sky

    @staticmethod
    def prepare_interpolated_data(weather_data, lat, lon):
        if not weather_data or "hourly" not in weather_data:
            return None

        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(weather_data["hourly"]["time"]),
                "temperature": weather_data["hourly"]["temperature_2m"],
                "wind_speed": weather_data["hourly"]["windspeed_10m"],
                "humidity": weather_data["hourly"]["relative_humidity_2m"],
                "rainfall": weather_data["hourly"]["rain"],
                "radiation": weather_data["hourly"]["shortwave_radiation"],
            }
        )

        df.set_index("datetime", inplace=True)

        clear_sky = WeatherData.calculate_clear_sky_irradiance(lat, lon, df.index)
        df["clear_sky_ghi"] = clear_sky["ghi"]

        df = df.resample("30T").interpolate()
        base_directory = os.getcwd()
        df.to_csv(base_directory + "/data/weather_conditions_NGED.csv")
        return df

    def fetch_weather(self, location_input, start_date, end_date):
        coords = location_input.split(",")

        if len(coords) == 2:  # If input is coordinates
            try:
                lat = float(coords[0])
                lon = float(coords[1])
                weather_data = self.get_weather_data(lat, lon, start_date, end_date)
                if weather_data:
                    return self.prepare_interpolated_data(weather_data, lat, lon)
                else:
                    raise ValueError("No weather data available.")
            except ValueError:
                raise ValueError("Invalid coordinates format. Use 'lat, lon'.")
        else:  # If input is a postcode
            lat, lon = self.get_coordinates_from_postcode(location_input)
            if lat is not None and lon is not None:
                weather_data = self.get_weather_data(lat, lon, start_date, end_date)
                if weather_data:
                    return self.prepare_interpolated_data(weather_data, lat, lon)
                else:
                    raise ValueError("No weather data available.")
            else:
                raise ValueError("Invalid postcode or unable to fetch data.")


# Example
if __name__ == "__main__":
    weather = WeatherData()
    try:
        location = (
            "52.769100, -1.556400"  # Replace with a valid UK postcode or coordinates
        )
        start_date = datetime.strptime("2024-01-01", "%Y-%m-%d")
        end_date = datetime.strptime("2026-08-01", "%Y-%m-%d")

        weather_df = weather.fetch_weather(location, start_date, end_date)
        print(weather_df)
    except ValueError as e:
        print(e)
