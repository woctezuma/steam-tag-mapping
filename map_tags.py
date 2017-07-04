# Objective: create a map of Steam tags
#
# Input:
#
# - a json file, downloaded from SteamSpy, named "steamspy.json"
json_filename = "steamspy.json"
steamspy_url = "http://steamspy.com/api.php?request=all"
# NB: If json_filename is missing, the current script will attempt to download and cache it from steamspy_url.

output_filename = "tag_matrix.txt"

import json

import numpy as np
from sklearn.manifold import TSNE

import pylab as pl
import math

import matplotlib.pyplot as plt
import matplotlib.colors as colors

with open(json_filename, 'r', encoding="utf8") as in_json_file:
    data = json.load(in_json_file)

tags = set()

for key in data.keys():
    current_tags = set(data[key]['tags'])
    tags = tags.union(current_tags)

str = "rogue"

for tag in tags:
    if str in tag:
        print(tag)

num_tags = len(tags)
print("#tags = %d" % num_tags)

tags_list = list(tags)
tags_list.sort()

tags_adjacency_matrix = np.zeros([num_tags, num_tags])
tag_counter_list = np.zeros(num_tags)

for key in data.keys():
    current_tags = list(data[key]['tags'])

    for index_i in range(len(current_tags)):
        i = tags_list.index(current_tags[index_i])
        # tags_adjacency_matrix[i][i] += 1
        tag_counter_list[i] += 1
        for index_j in range(index_i+1, len(current_tags)):
            j = tags_list.index(current_tags[index_j])
            tags_adjacency_matrix[i][j] += 1
            tags_adjacency_matrix[j][i] += 1

# Save the matrix to a text file
# np.savetxt(output_filename, tags_adjacency_matrix, fmt='%d', header=",".join(tags_list))

# tags_adjacency_matrix = np.loadtxt(output_filename)

model = TSNE(n_components=2, random_state=0)
X = model.fit_transform(tags_adjacency_matrix)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

chosen_tags_list = ["Visual Novel", "Anime", "VR", "Free to Play", "Rogue-lite", "Rogue-like", "Early Access", "Trading Card Game", "Card Game", "Sexual Content", "Gore", "Violent", "Nudity"]

isChosenTag = []
for tag in tags_list:
    if tag in chosen_tags_list:
        isChosenTag.append(True)
    else:
        isChosenTag.append(False)

# Scale and visualize the embedding vectors
def plot_embedding(X, str_list, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])

    delta = 0.001
    for i in range(X.shape[0]):
        my_color = "black"
        my_font_size = "xx-small"
        my_weight = 'ultralight'
        my_stretch = "ultra-condensed"
        if isChosenTag[i]:
            my_color = "red"
            my_font_size = "small"
            my_weight = 'normal'
            my_stretch = "condensed"
        plt.text(X[i, 0] + delta, X[i, 1] + delta, str_list[i], color=my_color, fontdict={'family': 'monospace', 'weight': my_weight, 'size': my_font_size, 'stretch': my_stretch})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

plot_embedding(X, tags_list, "Steam tags")

plt.show()