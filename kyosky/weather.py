# weather.py

# 1. First, sign up for an API key from OpenWeatherMap.
# 2. Install the necessary libraries using pip:
#    ```bash
#    pip install requests beautifulsoup4
#    ```

# 3. Create a Python script to fetch and display the weather data:

# ```python
# import requests

# # Replace 'YOUR_API_KEY' with your actual OpenWeatherMap API key
# api_key = 'YOUR_API_KEY'
# base_url = "http://api.openweathermap.org/data/2.5/weather?"

# # Coordinates for 47N 8E (Vienna, Austria)
# lat = 47.0690
# lon = 8.3015

# complete_url = base_url + "appid=" + api_key + "&lat=" + str(lat) + "&lon=" +
# str(lon)

# response = requests.get(complete_url)

# # Check if the request was successful
# if response.status_code == 200:
#     weather_data = response.json()

#     # Extracting data
#     main_weather = weather_data['weather'][0]['main']
#     description = weather_data['weather'][0]['description']
#     temperature = weather_data['main']['temp']

#     print(f"Current Weather at {lat}N {lon}E:")
#     print(f"Main: {main_weather}")
#     print(f"Description: {description}")
#     print(f"Temperature: {temperature} K")
# else:
#     print("Failed to retrieve data")

# # Displaying a weather symbol based on the main weather
# if 'Rain' in main_weather or 'rain' in description:
#     symbol = "🌧️"
# elif 'Snow' in main_weather or 'snow' in description:
#     symbol = "🌨️"
# elif 'Thunderstorm' in main_weather or 'thunderstorm' in description:
#     symbol = "⛈️"
# elif 'Mist' in main_weather or 'mist' in description:
#     symbol = "🌫️"
# elif 'Clear' in main_weather:
#     symbol = "☀️"
# else:
#     symbol = "❓"

# print(f"Current Weather Symbol: {symbol}")
# ```

# ### Explanation:
# 1. **API Key**: Replace `'YOUR_API_KEY'` with your actual OpenWeatherMap API key.
# 2. **Coordinates**: The coordinates for 47°N 8°E (Vienna, Austria) are used.
# 3. **Requests**: The script makes a GET request to the OpenWeatherMap API.
# 4. **Response Parsing**: It parses the JSON response to extract weather details
# such as main weather, description, and temperature.
# 5. **Weather Symbol**: Based on the main weather condition, it assigns an
# appropriate emoji symbol.

# ### Running the Script:
# - Save the script to a file, for example `weather.py`.
# - Run the script using Python:
#   ```bash
#   python weather.py
#   ```

# This will display the current weather details and the corresponding emoji symbol at
# your specified location.


import requests


# Function to fetch weather data from OpenWeatherMap
def fetch_weather(api_key, lat, lon):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&lat={lat}&lon={lon}"

    try:
        response = requests.get(complete_url)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.RequestException as e:
        print(f"Failed to retrieve data: {e}")
        return None


# Function to display weather details and symbol
def display_weather(weather_data):
    if not weather_data:
        return

    main_weather = weather_data["weather"][0]["main"]
    description = weather_data["weather"][0]["description"].lower()
    temperature = weather_data["main"]["temp"]

    print(f"Current Weather at 47N 8E:")
    print(f"Main: {main_weather}")
    print(f"Description: {description.capitalize()}")
    print(f"Temperature: {temperature} K")

    # Displaying a weather symbol based on the main weather
    if "rain" in description:
        symbol = "🌧️"
    elif "snow" in description:
        symbol = "🌨️"
    elif "thunderstorm" in description:
        symbol = "⛈️"
    elif "mist" in description or "haze" in description:
        symbol = "🌫️"
    elif "clear" in description:
        symbol = "☀️"
    else:
        symbol = "❓"

    print(f"Current Weather Symbol: {symbol}")


# Main function to execute the script
def main():
    # Replace 'YOUR_API_KEY' with your actual OpenWeatherMap API key
    api_key = "YOUR_API_KEY"

    # Coordinates for 47N 8E (Vienna, Austria)
    lat = 47.0690
    lon = 8.3015

    weather_data = fetch_weather(api_key, lat, lon)
    display_weather(weather_data)


# Execute the main function if this script is run directly
if __name__ == "__main__":
    main()
