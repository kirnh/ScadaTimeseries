import numpy as np
import json


def format_data(output_array, features_array, n_samples=None, timesteps_y=1, timesteps_x=96, randomize=True):
    x_list = []
    y_list = []
    n_points = len(output_array)
    if not n_samples:
        n_samples = n_points
    for i in range(n_samples):
        while True and randomize:
            i = np.random.randint(0, n_points)
            if i+timesteps_y+timesteps_x <= n_points:
                break
        # Stop once we can't have a y array of length timesteps_y
        if i+timesteps_y+timesteps_x > n_points:
            break
        x = features_array[:, i: i+timesteps_x].transpose()
        y = output_array[i+timesteps_x: i+timesteps_x+timesteps_y]
        x_list.append(x)
        y_list.append(y)
    x_list, y_list = np.array(x_list), np.array(y_list)
    return x_list.reshape((x_list.shape[0], timesteps_x, x_list.shape[2])), y_list.reshape((y_list.shape[0]))


def rescale_features(features):
    data_config = json.load(open("data_config.json"))
    feature_names = data_config["input_features"]
    feature_stats = json.load(open(data_config["data_dir"] + "/data_stats.json"))["features"]
    rescaled_features = []
    for i in range(len(features)):
        f = features[i]
        f_name = feature_names[i]
        f_stat = feature_stats[f_name]
        f_min = f_stat["min"]
        f_max = f_stat["max"]
        print(f_min)
        print(f_max)
        f_rescaled = f * (f_max-f_min) + f_min
        rescaled_features.append(f_rescaled)
    return rescaled_features, feature_names


