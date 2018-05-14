# Objective: download and cache data from SteamSpy

import json
import urllib.request


def download_steam_spy_data(json_filename="steamspy.json"):
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
            json_string = json.dumps(data)
            # Cache the json data to a local file
            with open(json_filename, 'w', encoding="utf8") as cache_json_file:
                print(json_string, file=cache_json_file)

    return data


def main():
    # noinspection PyUnusedLocal
    data = download_steam_spy_data()

    return True


if __name__ == "__main__":
    main()
