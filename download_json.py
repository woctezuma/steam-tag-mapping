# Objective: download and cache data from SteamSpy

import urllib.request, json

def downloadSteamSpyData(json_filename = "steamspy.json"):

    # If json_filename is missing, we will attempt to download and cache it from steamspy_url:
    steamspy_url = "http://steamspy.com/api.php?request=all"

    try:
        with open(json_filename, 'r', encoding="utf8") as in_json_file:
            data = json.load(in_json_file)
    except FileNotFoundError:
        print("Downloading and caching data from SteamSpy")
        # Trick to download the JSON file directly from SteamSpy, in case the file does not exist locally yet
        # Reference: https://stackoverflow.com/a/31758803/
        class AppURLopener(urllib.request.FancyURLopener):
            version = "Mozilla/5.0"
        opener = AppURLopener()
        with opener.open(steamspy_url) as response:
            data = json.load(response)
            # Make sure the json data is using double quotes instead of single quotes
            # Reference: https://stackoverflow.com/a/8710579/
            jsonString = json.dumps(data)
            # Cache the json data to a local file
            with open(json_filename, 'w', encoding="utf8") as cache_json_file:
                print(jsonString, file=cache_json_file)

    return data

if __name__ == "__main__":
    data = downloadSteamSpyData()