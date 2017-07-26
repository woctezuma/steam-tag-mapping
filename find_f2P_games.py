# Objective: find free-to-play Steam games which were previously in Early Access behind a paywall

from download_json import downloadSteamSpyData

base_steam_store_url = "http://store.steampowered.com/app/"

# SteamSpy's data in JSON format
data = downloadSteamSpyData()

num_games = len(data.keys())
print("#games = %d" % num_games)

# Tags to work with
tag_to_keep = "Free to Play"
tag_to_remove = "Early Access"

# Filter games based on tags
appid_list = []

# During peak:
ccu_threshold = pow(10,3)
num_hours_during_which_ccu_peak_reached = 0
# Off-peak:
num_players_off_peak = 0.01 * ccu_threshold
num_hours_off_peak = 24 - num_hours_during_which_ccu_peak_reached
# NB: We assume the ccu peak is reached for 5 hours a day, and the rest of the day, 10% of the peak is reached.
expected_num_players_daily = ccu_threshold * num_hours_during_which_ccu_peak_reached + num_players_off_peak * num_hours_off_peak
players_threshold = 14 * expected_num_players_daily

for appid in data.keys():
    current_tags = data[appid]['tags']
    if (tag_to_keep in current_tags) and not(tag_to_remove in current_tags):
        gamename = data[appid]['name'].lower().strip()
        if len(gamename)>1 and not("trial" in gamename):
            if data[appid]["players_2weeks"]>players_threshold and data[appid]["ccu"]>ccu_threshold:
                appid_list.append( int(appid) )

appid_list.sort()

# Display
dico = {}

for appid_int in appid_list:
    appid = str(appid_int)
    dico[appid] = data[appid]

counter = 0
for appid in sorted( dico.keys(), key=lambda x: dico[x]["players_2weeks"], reverse=True):
    counter += 1
    gamename = dico[appid]["name"]
    store_url = base_steam_store_url + str(appid)
    print(str(counter) + ".\t[" + gamename + "](" + store_url + ")")


num_games = len(appid_list)
print("#games = %d" % num_games)