from utils.ml import *
import os
from utils.data import *
from utils.validation import *

# ML configuration
ml_config = json.load(open("ml_config.json"))
model_name = ml_config["model_definition"]
weights_path = os.path.join(ml_config["weights_dir"], model_name, ml_config["weights_to_validate"])

validation_plot_path = os.path.join(ml_config["validation_plots_dir"], ml_config["model_definition"])
if not os.path.exists(validation_plot_path):
    os.makedirs(validation_plot_path)
validation_plot_path = validation_plot_path + "/validation_plot.png"

# Data configuration
data_config = json.load(open("data_config.json"))
data_stats = json.load(open(data_config["data_dir"] + "/data_stats.json"))
val_output_path = os.path.join(data_config["data_dir"], "val_output.npy")
val_features_path = os.path.join(data_config["data_dir"], "val_features.npy")

# Load model definition and weights
model = load_model_def(model_name)
model.load_weights(weights_path)

# Load validation data
val_x, val_y = format_data(np.load(val_output_path), np.load(val_features_path), randomize=False)
print("Validation x: ", val_x.shape)
print("Validation y: ", val_y.shape)

# Predict output using the trained model
val_y_pred = []
for i in range(len(val_y)):
    val_x_i = val_x[i]
    val_x_i = val_x_i.reshape((1, val_x_i.shape[0], val_x_i.shape[1]))
    y_pred = model.predict(val_x_i)
    val_y_pred.append(y_pred)

val_y_pred = np.array(val_y_pred)[:, 0, 0]
val_y_pred[val_y_pred < 0] = 0
val_y[val_y < 0] = 0

# Plot features, predicted output and measured output
val_features = np.load(val_features_path)
arrays = list(val_features)
array_labels = json.load(open("data_config.json"))["input_features"]

arrays += [val_y_pred, val_y]
array_labels += ["Prediction", "Ground truth"]

plot_graphs(arrays, array_labels, validation_plot_path)


