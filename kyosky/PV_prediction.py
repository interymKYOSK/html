import pandas as pd
import pvlib
from pvlib.pvsystem import PVSystem
import numpy as np
import matplotlib.pyplot as plt
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta

# Location parameters
lat = 48.09663828419101  # Kaiserstuhl
lon = 7.71013352029211
above_sea_level = 280

# Define the system parameters
tilt = 30  # degrees
azimuth = 180  # degrees, facing south
system_size = 600  # DC capacity of the system in watts (1000 = 1 kW)


def process_weather_data(df_weather, latitude, longitude, tilt, azimuth, system_size):
    # Convert time column to datetime format and ensure timezone awareness
    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"], utc=True)

    # Set location parameters
    location = pvlib.location.Location(latitude=latitude, longitude=longitude)

    # Solar position (get solar zenith and azimuth for each timestamp)
    solpos = location.get_solarposition(df_weather["datetime"])

    # Align solar position DataFrame to the weather DataFrame index
    solpos = solpos.set_index(df_weather.index)

    # Filter out times when the sun is below the horizon (apparent zenith > 90)
    df_daylight = df_weather[df_weather["ghi"] > 10].copy()

    # Recalculate solar position for the filtered data
    solpos_daylight = solpos.loc[df_daylight.index]

    # Calculate the total effective irradiance on the tilted surface for daylight hours only
    total_irradiance_daylight = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=df_daylight["dni"],
        ghi=df_daylight["ghi"],
        dhi=df_daylight["dhi"],
        solar_zenith=solpos_daylight["apparent_zenith"],
        solar_azimuth=solpos_daylight["azimuth"],
    )

    # Use the irradiance incident on the module plane for further calculations
    poa_irradiance_daylight = total_irradiance_daylight["poa_global"]

    # Calculate module temperature for daylight hours
    thermal_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
        "open_rack_glass_glass"
    ]
    temp_module_daylight = pvlib.temperature.sapm_cell(
        poa_global=poa_irradiance_daylight,
        temp_air=df_daylight["temp_air"],
        wind_speed=df_daylight["wind_speed"],
        **thermal_params,
    )

    # Define the PV system parameters
    module_parameters = {
        "pdc0": system_size,  # DC capacity of the system in watts
        "gamma_pdc": -0.004,  # Temperature coefficient of power in 1/C
    }
    inverter_parameters = {
        "pdc0": system_size,  # DC input to the inverter in watts
        "eta_inv_nom": 0.96,  # Nominal inverter efficiency
    }

    # Create a PVSystem object
    pv_system = pvlib.pvsystem.PVSystem(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        module_parameters=module_parameters,
        inverter_parameters=inverter_parameters,
    )

    # Calculate DC power output for daylight hours
    dc_power_daylight = pv_system.pvwatts_dc(
        g_poa_effective=poa_irradiance_daylight, temp_cell=temp_module_daylight
    )

    # Apply inverter efficiency to get AC power output
    inverter_efficiency = 0.96
    ac_power_daylight = dc_power_daylight * inverter_efficiency

    # Add AC power to the original DataFrame, filling in NaNs for nighttime
    df_weather["AC Power (kW)"] = np.nan
    df_weather.loc[df_daylight.index, "AC Power (kW)"] = (
        ac_power_daylight / 1000
    )  # convert W to kW

    # Fill NaNs with 0 for nighttime AC power
    df_weather.fillna(0, inplace=True)

    return df_weather


# Setup Open-Meteo client with caching and retry
cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Calculate date range for last 3 days
end_date = datetime.now().date()
start_date = end_date - timedelta(days=3)

print(f"Fetching historical data from {start_date} to {end_date}")

# Fetch historical data from Open-Meteo Archive API
archive_url = "https://archive-api.open-meteo.com/v1/archive"
archive_params = {
    "latitude": lat,
    "longitude": lon,
    "start_date": start_date.strftime("%Y-%m-%d"),
    "end_date": end_date.strftime("%Y-%m-%d"),
    "hourly": [
        "temperature_2m",
        "wind_speed_10m",
        "shortwave_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
    ],
    "wind_speed_unit": "ms",
}
archive_responses = openmeteo.weather_api(archive_url, params=archive_params)

# Process archive response
archive_response = archive_responses[0]
archive_hourly = archive_response.Hourly()

# Convert archive data into DataFrames
archive_hourly_data = {
    "datetime": pd.date_range(
        start=pd.to_datetime(archive_hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(archive_hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=archive_hourly.Interval()),
        inclusive="left",
    ),
    "temp_air": archive_hourly.Variables(0).ValuesAsNumpy(),
    "wind_speed": archive_hourly.Variables(1).ValuesAsNumpy(),
    "ghi": archive_hourly.Variables(2).ValuesAsNumpy(),
    "dhi": archive_hourly.Variables(3).ValuesAsNumpy(),
    "dni": archive_hourly.Variables(4).ValuesAsNumpy(),
}

df_archive_hourly = pd.DataFrame(archive_hourly_data)

# Process archive data - keep timezone aware
df_archive_hourly["datetime"] = pd.to_datetime(df_archive_hourly["datetime"], utc=True)
df_archive_hourly.set_index("datetime", inplace=True)

print(f"Historical data: {len(df_archive_hourly)} records")

# Fetch forecast data (DWD ICON)
print("Fetching forecast data")
forecast_url = "https://api.open-meteo.com/v1/forecast"
forecast_params = {
    "latitude": lat,
    "longitude": lon,
    "hourly": ["temperature_2m", "wind_speed_10m"],
    "minutely_15": [
        "shortwave_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
    ],
    "wind_speed_unit": "ms",
    "models": "icon_seamless",
}
forecast_responses = openmeteo.weather_api(forecast_url, params=forecast_params)

# Process forecast response
forecast_response = forecast_responses[0]
forecast_minutely_15 = forecast_response.Minutely15()
forecast_hourly = forecast_response.Hourly()

# Convert forecast data into DataFrames
forecast_minutely_15_data = {
    "datetime": pd.date_range(
        start=pd.to_datetime(forecast_minutely_15.Time(), unit="s", utc=True),
        end=pd.to_datetime(forecast_minutely_15.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=forecast_minutely_15.Interval()),
        inclusive="left",
    ),
    "dni": forecast_minutely_15.Variables(0).ValuesAsNumpy(),
    "ghi": forecast_minutely_15.Variables(1).ValuesAsNumpy(),
    "dhi": forecast_minutely_15.Variables(2).ValuesAsNumpy(),
}
forecast_hourly_data = {
    "datetime": pd.date_range(
        start=pd.to_datetime(forecast_hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(forecast_hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=forecast_hourly.Interval()),
        inclusive="left",
    ),
    "temp_air": forecast_hourly.Variables(0).ValuesAsNumpy(),
    "wind_speed": forecast_hourly.Variables(1).ValuesAsNumpy(),
}

df_forecast_minutely_15 = pd.DataFrame(forecast_minutely_15_data)
df_forecast_hourly = pd.DataFrame(forecast_hourly_data)

# Process forecast data - keep timezone aware
df_forecast_hourly["datetime"] = pd.to_datetime(
    df_forecast_hourly["datetime"], utc=True
)
df_forecast_hourly.set_index("datetime", inplace=True)
df_forecast_minutely_15["datetime"] = pd.to_datetime(
    df_forecast_minutely_15["datetime"], utc=True
)
df_forecast_minutely_15.set_index("datetime", inplace=True)

# Resample forecast hourly data
df_forecast_hourly_resampled = df_forecast_hourly.resample("15T").interpolate()

# Merge forecast data
df_weather_forecast = df_forecast_minutely_15.merge(
    df_forecast_hourly_resampled, left_index=True, right_index=True, how="left"
)
df_weather_forecast.reset_index(inplace=True)

print(f"Forecast data: {len(df_weather_forecast)} records")

# Reset index for archive data to match forecast structure
df_archive_hourly_reset = df_archive_hourly.reset_index()

# Combine historical and forecast data
df_weather_combined = pd.concat(
    [df_archive_hourly_reset, df_weather_forecast], ignore_index=True
)
df_weather_combined = (
    df_weather_combined.drop_duplicates(subset=["datetime"])
    .sort_values("datetime")
    .reset_index(drop=True)
)

print(f"Combined data: {len(df_weather_combined)} records")
print(df_weather_combined.head())

# Process combined weather data
df_processed = process_weather_data(
    df_weather_combined, lat, lon, tilt, azimuth, system_size
)

# Create a figure with multiple subplots
fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

# Plot GHI
axs[0, 0].plot(
    df_processed["datetime"],
    df_processed["ghi"],
    label="GHI",
    color="orange",
)
axs[0, 0].axvline(
    x=pd.Timestamp.now(tz="UTC"), color="red", linestyle="--", alpha=0.7, label="Now"
)
axs[0, 0].set_title("Global Horizontal Irradiance (GHI)")
axs[0, 0].set_ylabel("GHI (W/m²)")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot DNI
axs[0, 1].plot(
    df_processed["datetime"],
    df_processed["dni"],
    label="DNI",
    color="orange",
)
axs[0, 1].axvline(
    x=pd.Timestamp.now(tz="UTC"), color="red", linestyle="--", alpha=0.7, label="Now"
)
axs[0, 1].set_title("Direct Normal Irradiance (DNI)")
axs[0, 1].set_ylabel("DNI (W/m²)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot DHI
axs[1, 0].plot(
    df_processed["datetime"],
    df_processed["dhi"],
    label="DHI",
    color="orange",
)
axs[1, 0].axvline(
    x=pd.Timestamp.now(tz="UTC"), color="red", linestyle="--", alpha=0.7, label="Now"
)
axs[1, 0].set_title("Diffuse Horizontal Irradiance (DHI)")
axs[1, 0].set_ylabel("DHI (W/m²)")
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plot Temperature
axs[1, 1].plot(
    df_processed["datetime"],
    df_processed["temp_air"],
    label="Temperature",
    color="orange",
)
axs[1, 1].axvline(
    x=pd.Timestamp.now(tz="UTC"), color="red", linestyle="--", alpha=0.7, label="Now"
)
axs[1, 1].set_title("Air Temperature")
axs[1, 1].set_ylabel("Temp (°C)")
axs[1, 1].legend()
axs[1, 1].grid(True)

# Plot Wind Speed
axs[2, 0].plot(
    df_processed["datetime"],
    df_processed["wind_speed"],
    label="Wind Speed",
    color="orange",
)
axs[2, 0].axvline(
    x=pd.Timestamp.now(tz="UTC"), color="red", linestyle="--", alpha=0.7, label="Now"
)
axs[2, 0].set_title("Wind Speed")
axs[2, 0].set_ylabel("Wind Speed (m/s)")
axs[2, 0].legend()
axs[2, 0].grid(True)

# Plot AC Power
axs[2, 1].plot(
    df_processed["datetime"],
    df_processed["AC Power (kW)"],
    label="AC Power",
    color="orange",
)
axs[2, 1].axvline(
    x=pd.Timestamp.now(tz="UTC"), color="red", linestyle="--", alpha=0.7, label="Now"
)
axs[2, 1].set_title("AC Power Output")
axs[2, 1].set_ylabel("AC Power (kW)")
axs[2, 1].legend()
axs[2, 1].grid(True)

# Set the x-axis label for the bottom row
for ax in axs[2, :]:
    ax.set_xlabel("Datetime")
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===")
total_energy = df_processed["AC Power (kW)"].sum() * 0.25  # kWh (15-min intervals)
print(f"Total energy over period: {total_energy:.2f} kWh")
print(
    f"Average power during daylight: {df_processed[df_processed['AC Power (kW)'] > 0]['AC Power (kW)'].mean():.3f} kW"
)
print(f"Peak power: {df_processed['AC Power (kW)'].max():.3f} kW")
