import os
from pandas import DataFrame as df
import numpy as np
import json
from sklearn.preprocessing import minmax_scale

data_config = json.load(open("data_config.json", "r"))


def get_dataframe():
    csv_dir = data_config["csv_dir"]
    data_initialized = False
    for f in os.listdir(csv_dir):
        csv_filepath = os.path.join(csv_dir, f)
        new_data = df.from_csv(csv_filepath, encoding="ISO-8859-1", sep=';')
        new_data.fillna(data_config["nan_replace"])
        if not data_initialized:
            data = new_data
            data_initialized = True
        else:
            data = data.append(new_data)
    return data.sort_index()


def clean_feature(dataframe, feature_name, valid_time_period=None, invalid_time_value=0):
    feature = dataframe[feature_name]
    if valid_time_period:
        times_list = [i.hour for i in feature.index.time]
        invalid_times = [not(valid_time_period[1] > i > valid_time_period[0]) for i in times_list]
        feature.loc[invalid_times] = invalid_time_value
    dataframe[feature_name] = feature
    return dataframe


def split_and_save_data(dataframe, save_data_stats=True):
    data_dir = data_config["data_dir"]
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_stats = {}
    # Clean all features
    feature_names = data_config["input_features"]
    for feature_name in feature_names:
        if feature_name not in data_config["feature_cleaning"]:
            continue
        clean_settings = data_config["feature_cleaning"][feature_name]
        valid_time = clean_settings["valid_time"]
        invalid_time_value = clean_settings["invalid_time_value"]
        dataframe = clean_feature(dataframe, feature_name, valid_time_period=valid_time, invalid_time_value=invalid_time_value)

    output = dataframe[data_config["output_feature"]]

    o_max = output.max()
    o_min = output.min()
    o_mean = output.mean()
    data_stats["output"] = {"min": o_min,
                            "max": o_max,
                            "mean": o_mean}
    output = minmax_scale(output)
    output[np.isnan(output)] = data_config["nan_replace"]

    feature_stats = {}
    features = []
    for f in feature_names:
        feature = dataframe[f]
        f_min = feature.min()
        f_max = feature.max()
        f_mean = feature.mean()
        feature = minmax_scale(feature)
        feature[np.isnan(feature)] = data_config["nan_replace"]
        features.append(feature)
        feature_stats[f] = {"min": f_min,
                            "max": f_max,
                            "mean": f_mean}
    features = np.array(features)
    data_stats["features"] = feature_stats

    n_points = output.shape[0]
    print("[INFO] Number of data points = {}".format(n_points))
    print("[INFO] Number of features = {}".format(features.shape[0]))
    idx_1 = int(n_points * data_config["train_test_val_split"][0])
    idx_2 = int(idx_1 + n_points * data_config["train_test_val_split"][1])

    train_output, train_features = output[:idx_1], features[:, :idx_1]
    test_output, test_features = output[idx_1: idx_2], features[:, idx_1: idx_2]
    val_output, val_features = output[idx_2:], features[:, idx_2:]

    print("[INFO] Number of train points: {}".format(len(train_output)))
    print("[INFO] Number of test points: {}".format(len(test_output)))
    print("[INFO] Number of validation points: {}".format(len(val_output)))

    np.save(os.path.join(data_dir, "train_output.npy"), train_output)
    np.save(os.path.join(data_dir, "train_features.npy"), train_features)
    np.save(os.path.join(data_dir, "test_output.npy"), test_output)
    np.save(os.path.join(data_dir, "test_features.npy"), test_features)
    np.save(os.path.join(data_dir, "val_output.npy"), val_output)
    np.save(os.path.join(data_dir, "val_features.npy"), val_features)

    if save_data_stats:
        print("[INFO] Saving data statistics...")
        json.dump(data_stats, open(os.path.join(data_dir, "data_stats.json"), "w"))


if __name__ == '__main__':
    data = get_dataframe()
    split_and_save_data(data)
