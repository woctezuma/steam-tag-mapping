# Objective: create a map of Steam tags

import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from download_json import downloadSteamSpyData

# SteamSpy's data in JSON format
data = downloadSteamSpyData()

num_games = len(data.keys())
print("#games = %d" % num_games)

# Create a set of all Steam tags
tags = set()

for appid in data.keys():
    current_tags = set(data[appid]['tags'])
    tags = tags.union(current_tags)

num_tags = len(tags)
print("#tags = %d" % num_tags)

# Print tags containing the word "rogue"
wordToSearch = "rogue"
wordToSearch = wordToSearch.lower()

for tag in tags:
    tag_str = str(tag).lower()
    if wordToSearch in tag_str:
        print(tag)

# Create a list of tags sorted in lexicographical order
tags_list = list(tags)
tags_list.sort()

# Define a list of tags to display in bold
chosen_tags_set = set(["Visual Novel", "Anime", "VR", "Free to Play", "Rogue-lite", "Rogue-like", "Early Access",
                       "Trading Card Game", "Card Game", "Gore", "Violent", "Sexual Content", "Nudity"])

for tag in chosen_tags_set.difference(tags):
    print("Tag " + tag + " is not used for any game.")

chosen_tags_set = chosen_tags_set.intersection(tags)

isChosenTag = [bool(tag in chosen_tags_set) for tag in tags_list]

# Create an adjacency matrix (symmetric with zeros on the diagonal)
tags_adjacency_matrix_filename = "tags_adjacency_matrix.txt"

try:
    # Load the matrix from a text file
    tags_adjacency_matrix = np.loadtxt(tags_adjacency_matrix_filename)

except FileNotFoundError:
    tags_adjacency_matrix = np.zeros([num_tags, num_tags])

    for appid in data.keys():
        current_tags = list(data[appid]['tags'])

        for index_i in range(len(current_tags)):
            i = tags_list.index(current_tags[index_i])

            for index_j in range(index_i+1, len(current_tags)):
                j = tags_list.index(current_tags[index_j])

                tags_adjacency_matrix[i][j] += 1
                tags_adjacency_matrix[j][i] += 1

    # Save the matrix to a text file
    np.savetxt(tags_adjacency_matrix_filename, tags_adjacency_matrix, fmt='%d', header=",".join(tags_list))

# Create tag-joint-game matrix (tags in lines, games in columns)
tag_joint_game_matrix_filename = "tag_joint_game_matrix.txt"

try:
    # Load the matrix from a text file
    tag_joint_game_matrix = np.loadtxt(tag_joint_game_matrix_filename)

except FileNotFoundError:
    tag_joint_game_matrix = np.zeros([num_tags, num_games])

    game_counter = 0

    for appid in data.keys():
        current_tags = list(data[appid]['tags'])

        for tag in current_tags:
            i = tags_list.index(tag)
            j = game_counter
            tag_joint_game_matrix[i][j] += 1

        game_counter += 1

    # Save the matrix to a text file
    np.savetxt(tag_joint_game_matrix_filename, tag_joint_game_matrix, fmt='%d')

# Compute the mapping ussing t-SNE
# Reference: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
model = TSNE(n_components=2, random_state=0)
X = model.fit_transform(tags_adjacency_matrix)

# Scale and visualize the embedding vectors
def plot_embedding(X, str_list, title=None, delta_font=pow(10, -3)):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()

    plt.scatter(X[:, 0], X[:, 1])

    for i in range(X.shape[0]):

        if isChosenTag[i]:
            my_color = "red"
            my_font_size = "small"
            my_weight = 'normal'
            my_stretch = "condensed"
        else:
            my_color = "black"
            my_font_size = "xx-small"
            my_weight = 'ultralight'
            my_stretch = "ultra-condensed"

        plt.text(X[i, 0] + delta_font, X[i, 1] + delta_font, str_list[i], color=my_color,
                 fontdict={'family': 'monospace', 'weight': my_weight, 'size': my_font_size, 'stretch': my_stretch})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()

# Display
my_title = "Steam tags"
plot_embedding(X, tags_list, my_title)
