# Objective: create a map of Steam tags

import matplotlib

matplotlib.use('Agg')

import steamspypi
import json

# noinspection PyPep8
import matplotlib.pyplot as plt
# noinspection PyPep8
import numpy as np
# noinspection PyPep8
from sklearn.decomposition import TruncatedSVD
# noinspection PyPep8
from sklearn.manifold import TSNE


def get_all_tags(data):
    # Create a set of all Steam tags
    tags = set()

    for appid in data.keys():
        current_tags = set(data[appid]['tags'])
        tags = tags.union(current_tags)

    num_tags = len(tags)
    print("#tags = %d" % num_tags)

    return tags


def display_tags_containing_specific_word(tags, word_to_search='rogue'):
    # Print tags containing the word "rogue"

    word_to_search = word_to_search.lower()

    for tag in tags:
        tag_str = str(tag).lower()
        if word_to_search in tag_str:
            print(tag)

    return


def filter_chosen_tags(chosen_tags_set, tags):
    for tag in chosen_tags_set.difference(tags):
        print("Tag " + tag + " is not used for any game.")

    chosen_tags_set = chosen_tags_set.intersection(tags)

    return chosen_tags_set


def get_adjacency_matrix(data, tags):
    # Create an adjacency matrix (symmetric with zeros on the diagonal)
    tags_adjacency_matrix_filename = "tags_adjacency_matrix.txt"
    tags_counter_filename = "tags_counter.txt"

    try:
        # Load the matrix from a text file
        tags_adjacency_matrix = np.loadtxt(tags_adjacency_matrix_filename)

        # Load the counter list from a text file
        tags_counter = np.loadtxt(tags_counter_filename)

    except FileNotFoundError:
        num_tags = len(tags)

        # Create a list of tags sorted in lexicographical order
        tags_list = list(tags)
        tags_list.sort()

        tags_adjacency_matrix = np.zeros([num_tags, num_tags])
        tags_counter = np.zeros(num_tags)

        for appid in data.keys():
            current_tags = list(data[appid]['tags'])

            for index_i in range(len(current_tags)):
                i = tags_list.index(current_tags[index_i])
                tags_counter[i] += 1

                for index_j in range(index_i + 1, len(current_tags)):
                    j = tags_list.index(current_tags[index_j])

                    tags_adjacency_matrix[i][j] += 1
                    tags_adjacency_matrix[j][i] += 1

        # Save the matrix to a text file
        # noinspection PyTypeChecker
        np.savetxt(tags_adjacency_matrix_filename, tags_adjacency_matrix, fmt='%d', header=",".join(tags_list))
        # Save the counter list to a text file
        # noinspection PyTypeChecker
        np.savetxt(tags_counter_filename, tags_counter, fmt='%d', header=",".join(tags_list))

    # Normalize the pairwise similarity matrix, but only after the text file was saved so that int are saved, not float.
    # Reference: "Can I use a pairwise similarity matrix as input into t-SNE?" in http://lvdmaaten.github.io/tsne/
    tags_adjacency_matrix /= tags_adjacency_matrix.sum()

    return tags_adjacency_matrix, tags_counter


def get_sorted_tags(tags):
    # Create a list of tags sorted in lexicographical order
    tags_list = list(tags)
    tags_list.sort()

    return tags_list


def get_tag_joint_game_matrix(data, tags):
    # Create tag-joint-game matrix (tags in lines, games in columns)
    tag_joint_game_matrix_filename = "tag_joint_game_matrix.txt"

    try:
        # Load the matrix from a text file
        tag_joint_game_matrix = np.loadtxt(tag_joint_game_matrix_filename)

    except FileNotFoundError:
        num_games = len(data.keys())
        num_tags = len(tags)

        tags_list = get_sorted_tags(tags)

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
        # noinspection PyTypeChecker
        np.savetxt(tag_joint_game_matrix_filename, tag_joint_game_matrix, fmt='%d')

    return tag_joint_game_matrix


def compute_tsne_mapping_of_steam_tags(tags_adjacency_matrix, tag_joint_game_matrix, use_data_directly_as_input=True):
    # Compute the mapping of Steam tags using t-SNE
    # Reference: http://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne

    num_components_svd = 50
    num_components_tsne = 2

    svd = TruncatedSVD(n_components=num_components_svd, random_state=0)

    # We have chosen a learning rate lower than the default (1000) so that the error decreases during early iterations:
    tsne = TSNE(n_components=num_components_tsne, random_state=0, verbose=2, learning_rate=400, perplexity=25)

    if use_data_directly_as_input:
        # Either directly use the matrix joining tag and game, in 2 steps:

        # 1st step: reduce the dimensionality of the input SPARSE matrix, with TruncatedSVD as suggested in:
        # http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        reduced_matrix = svd.fit_transform(tag_joint_game_matrix)

        # 2nd step: apply t-SNE to the reduced DENSE matrix
        # noinspection PyPep8Naming
        X = tsne.fit_transform(reduced_matrix)

    else:
        # Or use the pairwise similarity matrix (yes, you can do that too):
        # Reference: http://lvdmaaten.github.io/tsne/
        # noinspection PyPep8Naming
        X = tsne.fit_transform(tags_adjacency_matrix)

    return X


# Scale and visualize the embedding vectors
# noinspection PyPep8Naming
def plot_embedding(X, str_list, chosen_tags_set, title=None, delta_font=0.003):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    # noinspection PyPep8Naming
    X = (X - x_min) / (x_max - x_min)

    plt.figure()

    plt.scatter(X[:, 0], X[:, 1])

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    # References:
    # * https://stackoverflow.com/a/40729950/
    # * http://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html
    for index, (label, x, y) in enumerate(
            zip(str_list, X[:, 0], X[:, 1])):

        dx = x - X[:, 0]
        dx[index] = 1
        dy = y - X[:, 1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + delta_font
        else:
            horizontalalignment = 'right'
            x = x - delta_font
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + delta_font
        else:
            verticalalignment = 'top'
            y = y - delta_font

        if label in chosen_tags_set:
            my_color = "red"
        else:
            my_color = "black"

        my_font_size = "medium"
        my_weight = 'normal'
        my_stretch = "condensed"

        plt.text(x, y, label, color=my_color,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 fontdict={'family': 'monospace', 'weight': my_weight, 'size': my_font_size, 'stretch': my_stretch})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()


# noinspection PyPep8Naming
def optimize_display(X, chosen_tags_set, tags, tags_adjacency_matrix, tags_counter, perform_trimming):
    # Trim the data based on different counters, for better display

    num_tags = len(tags)

    tags_list = get_sorted_tags(tags)

    tags_counter /= tags_counter.sum()
    assert (len(tags_counter) == num_tags)
    # NB: tags_counter gives the #occurences of each tag

    links_counter = np.sum(tags_adjacency_matrix, axis=1) / tags_adjacency_matrix.sum()
    assert (len(links_counter) == num_tags)
    # NB: links_counter_list gives the number of links between a tag and every other given tag

    # Aggregate overall statistics regarding tags
    tags_statistics = [(i, j, k) for (i, j, k) in zip(tags, tags_counter, links_counter)]

    # Compute percentiles

    prct = 7
    while np.percentile(tags_counter, prct) > np.min(
            [tags_counter[i] for i in range(len(tags_list)) if tags_list[i] in chosen_tags_set]):
        prct -= 1
    prct -= 1
    low_q = np.percentile(tags_counter, prct)
    print("Low percentile for %d" % prct)

    prct = 90
    while np.percentile(tags_counter, prct) < np.max(
            [tags_counter[i] for i in range(len(tags_list)) if tags_list[i] in chosen_tags_set]):
        prct += 1
    prct += 1
    high_q = np.percentile(tags_counter, prct)
    print("High percentile for %d" % prct)

    common_tags = [v[0] for v in tags_statistics if bool(v[1] <= low_q)]
    rare_tags = [v[0] for v in tags_statistics if bool(v[1] >= high_q)]
    is_tag_good = [not ((tag in common_tags) or (tag in rare_tags)) for tag in tags_list]

    # Perform the trimming

    if perform_trimming:
        # noinspection PyPep8Naming
        X_trimmed = np.array([list(X[val, :]) for is_good, val in zip(is_tag_good, range(X.shape[0])) if is_good])
        tags_list_trimmed = [val for is_good, val in zip(is_tag_good, tags_list) if is_good]
    else:
        # noinspection PyPep8Naming
        X_trimmed = X
        tags_list_trimmed = tags_list

    # Display
    my_title = "Map of Steam tags"
    plot_embedding(X_trimmed, tags_list_trimmed, chosen_tags_set, my_title)

    return


def main():
    # Whether to map tags based on the input data directly, or based on an intermediate step with a similarity matrix
    use_data_directly_as_input = True

    # Boolean to decide whether to trim out the most common and most rare tags when displaying the map
    perform_trimming = True

    # SteamSpy's data in JSON format
    data = steamspypi.load()

    try:
        tags = get_all_tags(data)
    except KeyError:
        # SteamSpy API does not provide tags anymore, so load a database downloaded from the SteamSpy ages ago.
        with open('steamspy.json', 'r', encoding='utf8') as f:
            data = json.load(f)

        tags = get_all_tags(data)

    num_games = len(data.keys())
    print("#games = %d" % num_games)

    word_to_search = 'rogue'
    display_tags_containing_specific_word(tags, word_to_search)

    # Define a list of tags to display in bold
    chosen_tags_set = {"Visual Novel", "Anime", "VR", "Free to Play", "Rogue-lite", "Rogue-like", "Early Access",
                       "Trading Card Game", "Card Game", "Gore", "Violent", "Sexual Content", "Nudity"}

    chosen_tags_set = filter_chosen_tags(chosen_tags_set, tags)

    tags_adjacency_matrix, tags_counter = get_adjacency_matrix(data, tags)

    if use_data_directly_as_input:
        tag_joint_game_matrix = get_tag_joint_game_matrix(data, tags)
    else:
        tag_joint_game_matrix = None

    # noinspection PyPep8Naming
    X = compute_tsne_mapping_of_steam_tags(tags_adjacency_matrix, tag_joint_game_matrix, use_data_directly_as_input)

    optimize_display(X, chosen_tags_set, tags, tags_adjacency_matrix, tags_counter, perform_trimming)

    return True


if __name__ == '__main__':
    main()
