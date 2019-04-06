# Steam Tag Mapping

 [![Build status][Build image]][Build] [![Updates][Dependency image]][PyUp] [![Python 3][Python3 image]][PyUp] [![Code coverage][Codecov image]][Codecov]

  [Build]: https://travis-ci.org/woctezuma/steam-tag-mapping
  [Build image]: https://travis-ci.org/woctezuma/steam-tag-mapping.svg?branch=master

  [PyUp]: https://pyup.io/repos/github/woctezuma/steam-tag-mapping/
  [Dependency image]: https://pyup.io/repos/github/woctezuma/steam-tag-mapping/shield.svg
  [Python3 image]: https://pyup.io/repos/github/woctezuma/steam-tag-mapping/python-3-shield.svg

  [Codecov]: https://codecov.io/gh/woctezuma/steam-tag-mapping
  [Codecov image]: https://codecov.io/gh/woctezuma/steam-tag-mapping/branch/master/graph/badge.svg

This repository allows to compute an embedding of Steam tags, based on all Steam games and their respective tags.

## Definition ##

A "tag" is a word or expression attached to a game, as can be seen [on Steam](http://store.steampowered.com//tag/browse).

## Data source ##

The data comes from [SteamSpy API](http://steamspy.com/api.php).

To run the code, you will need:
* data from SteamSpy: `steamspy.json` (automatically downloaded if the file is missing)

The data is included along the code in this repository, as downloaded on June 30, 2017.

## Requirements ##

- Install the latest version of [Python 3.X](https://www.python.org/downloads/).

- Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage ##

To create the mapping by analyzing the joint occurences of tags for each game, run the Python script as follows:
```bash
python map_tags.py
```

## Results ##
* [a NeoGAF post](http://www.neogaf.com/forum/showpost.php?p=242575674&postcount=7426) showing the map of tags.

Overview:
![Map of Steam tags](http://i.imgur.com/O2vwzoy.png "Map of Steam tags")

Zoom in:
![Zoom on Visual Novel](http://i.imgur.com/tD5yZQ7.png "Zoom on Visual Novel")
