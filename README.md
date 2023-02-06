# Steam Tag Mapping

[![Build status][build-image]][build]
[![Code coverage][codecov-image]][codecov]
[![Code Quality][codacy-image]][codacy]

This repository allows to compute an embedding of Steam tags, based on all Steam games and their respective tags.

![Map of Steam tags](https://github.com/woctezuma/steam-tag-mapping/wiki/img/QuUcF85.png)

## Definition

A "tag" is a word or expression attached to a game, as can be seen [on Steam](http://store.steampowered.com//tag/browse).

## Data source

Data is downloaded from [SteamSpy API](http://steamspy.com/api.php) via the [steamtags](https://github.com/woctezuma/match-steam-tags) PyPI package.

## Requirements

-   Install the latest version of [Python 3.X](https://www.python.org/downloads/).
-   Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To create the mapping by analyzing the joint occurences of tags for each game, run the Python script as follows:

```bash
python map_tags.py
```

## Results

Results are shown [on the Wiki](https://github.com/woctezuma/steam-tag-mapping/wiki).

## Reference

-   [a post on NeoGAF](http://www.neogaf.com/forum/showpost.php?p=242575674&postcount=7426) showing the map of tags.
-   t-SNE:
    - [author's FAQ][tsne-author]
    - [wikipedia][tsne-wiki]
-   [UMAP][umap-code]

<!-- Definitions -->

[build]: <https://github.com/woctezuma/steam-tag-mapping/actions>
[build-image]: <https://github.com/woctezuma/steam-tag-mapping/workflows/Python package/badge.svg?branch=master>

[pyup]: <https://pyup.io/repos/github/woctezuma/steam-tag-mapping/>
[dependency-image]: <https://pyup.io/repos/github/woctezuma/steam-tag-mapping/shield.svg>
[python3-image]: <https://pyup.io/repos/github/woctezuma/steam-tag-mapping/python-3-shield.svg>

[codecov]: <https://codecov.io/gh/woctezuma/steam-tag-mapping>
[codecov-image]: <https://codecov.io/gh/woctezuma/steam-tag-mapping/branch/master/graph/badge.svg>

[codacy]: <https://www.codacy.com/app/woctezuma/steam-tag-mapping>
[codacy-image]: <https://api.codacy.com/project/badge/Grade/ea42bcc3210b442cbc40e4b0e9e016d2>

[tsne-author]: <https://lvdmaaten.github.io/tsne/>
[tsne-wiki]: <https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding>
[umap-code]: <https://github.com/lmcinnes/umap>
