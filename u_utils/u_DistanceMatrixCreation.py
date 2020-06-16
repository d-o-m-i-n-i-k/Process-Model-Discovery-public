import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import csv
import copy
import z_setting_parameters as settings


def get_distance_matrix():
    # get paths
    adjacency_matrix_path = settings.path_data_sources + settings.filename_adjacency_matrix
    room_dict_path = settings.path_data_sources + settings.filename_room_separation

    adjacency_matrix, sensor_labels = import_adjacency(adjacency_matrix_path)

    distance_matrix = floyd_warshall(copy.deepcopy(adjacency_matrix))

    room_dict = import_rooms(room_dict_path)

    # get a mapping where each sensor has their room label
    sensor_to_room_dict = {}
    for key, value in room_dict.items():
        for string in value:
            sensor_to_room_dict.setdefault(string, []).append(key)

    # add support for zeros
    # convert to numpy array
    distance_matrix = np.array(distance_matrix)

    # zero has zero distance to other zeros
    distance_matrix[0][0] = 0

    dict_distance_adjacency_sensor = {'adjacency_matrix': adjacency_matrix,
                                      'distance_matrix': distance_matrix,
                                      'room_dict': room_dict,
                                      'sensor_labels': sensor_labels,
                                      'sensor_to_room_dict': sensor_to_room_dict}

    pd.DataFrame(distance_matrix).to_csv(path_or_buf=settings.path_data_sources + 'distance_matrix.csv',
                                         sep=';')
    return dict_distance_adjacency_sensor


def set_zero_distance_value(distance_matrix, zero_distance_value):
    # overwrite zero_distance_value to first column
    distance_matrix[:, 0] = zero_distance_value
    # overwrite zero_distance_value to first row
    distance_matrix[0, :] = zero_distance_value

    return distance_matrix


def import_adjacency(csv_file_name):
    data = np.genfromtxt(csv_file_name,
                         delimiter=';',
                         filling_values=0,
                         skip_header=1,
                         dtype=int)

    labels = np.genfromtxt(csv_file_name,
                           delimiter=';',
                           skip_header=0,
                           dtype=str,
                           max_rows=1)

    # delete the first column, because it contains only the sensor label
    labels = labels[1:]
    data = data[:, 1:]

    return data, labels


def import_rooms(csv_file_name):
    """
    Creates a dictionary for the rooms in the smart-home
    :param csv_file_name: File-Path to CSV-File
    :return: dictionary with rooms as keys and sensor labels as values
    """
    data_dict = {}
    with open(csv_file_name, mode='r') as infile:
        reader = csv.reader(infile, delimiter=';')
        for row in reader:
            data_dict[row[0]] = [int(element) for element in row[1:] if element != '']

    return data_dict


def floyd_warshall(adj_matrix):
    """ dist[][] will be the output matrix that will finally
        have the shortest distances between every pair of vertices """
    """ initializing the solution matrix same as input graph matrix 
    OR we can say that the initial values of shortest distances 
    are based on shortest paths considering no  
    intermediate vertices """
    # dist = map(lambda i: map(lambda j: j, i), adj_matrix)
    dist = adj_matrix
    for n in range(0, len(dist)):
        for m in range(0, len(dist[n])):
            if dist[n][m] == 0:
                dist[n][m] = 999999
            if n == m:
                dist[n][m] = 0
    """ Add all vertices one by one to the set of intermediate 
     vertices. 
     ---> Before start of an iteration, we have shortest distances 
     between all pairs of vertices such that the shortest 
     distances consider only the vertices in the set  
    {0, 1, 2, .. k-1} as intermediate vertices. 
      ----> After the end of an iteration, vertex no. k is 
     added to the set of intermediate vertices and the  
    set becomes {0, 1, 2, .. k} 
    """
    for k in range(len(adj_matrix)):

        # pick all vertices as source one by one
        for i in range(len(adj_matrix)):

            # Pick all vertices as destination for the
            # above picked source
            for j in range(len(adj_matrix)):
                # If vertex k is on the shortest path from
                # i to j, then update the value of dist[i][j]
                dist[i][j] = min(dist[i][j],
                                 dist[i][k] + dist[k][j])
    return dist


def draw_adjacency_graph(dict_room_information,
                         data_sources_path,
                         filename_adjacency_plot):
    adjacency_matrix = dict_room_information['adjacency_matrix']
    sensor_labels = dict_room_information['sensor_labels']
    room_dict = dict_room_information['room_dict']

    # list of available colours to be able to loop through
    available_colors = ['lightgreen', 'orange', 'crimson', 'darkgreen', 'grey', 'yellow', 'pink', 'blue',
                        'cyan', 'lime', 'maroon', 'green', 'red', 'brown', 'lavender', 'navy', 'salmon', 'ivory']

    # empty color list
    color_map = [None] * len(sensor_labels)
    iterator = 0
    legend = []
    for room in room_dict:
        for sensor in room_dict[room]:
            color_map[sensor] = available_colors[iterator]

        legend.append((room, available_colors[iterator]))
        # change colour if moving on to next room
        if iterator == len(available_colors) - 1:
            iterator = 0
        else:
            iterator += 1

    # set figure size
    plot = plt.figure(figsize=(15, 15))
    my_labels_mapping = dict(enumerate(sensor_labels, start=0))

    graph_object_data = nx.relabel_nodes(nx.from_numpy_matrix(adjacency_matrix), my_labels_mapping)

    # increase spacing between nodes
    pos = nx.spring_layout(graph_object_data, k=0.85 * 1 / np.sqrt(len(graph_object_data.nodes())), iterations=20)

    # the following line throws a deprecation warning
    # nx.draw(graph_object_data, node_color=color_map, with_labels=True, node_size=1000, pos=pos)
    nx.draw(graph_object_data, with_labels=True, node_size=1000, pos=pos)
    # make legend
    patches = []
    for element in legend:
        patches.append(mpatches.Patch(color=element[1], label=element[0]))
    plt.legend(handles=patches)
    plt.show()
    export_file_name_path = data_sources_path + filename_adjacency_plot
    plot.savefig(export_file_name_path)
