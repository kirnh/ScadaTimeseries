from utils.data import format_data
from utils.ml import *
from utils.callbacks import *
import os
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.optimizers

# Data configuration
data_config = json.load(open("data_config.json"))
data_dir = data_config["data_dir"]
timesteps_x = data_config["input_timesteps"]
timesteps_y = data_config["output_timesteps"]
n_train_samples = data_config["n_train_samples"]
n_test_samples = data_config["n_test_samples"]
train_output_path = os.path.join(data_dir, "train_output.npy")
train_features_path = os.path.join(data_dir, "train_features.npy")
test_output_path = os.path.join(data_dir, "test_output.npy")
test_features_path = os.path.join(data_dir, "test_features.npy")

# ML configuration
ml_config = json.load(open("ml_config.json"))
model_name = ml_config["model_definition"]
model = load_model_def(model_name)
if ml_config["experiment_slug"]:
    model_name += ml_config["experiment_slug"]

weights_dir = ml_config["weights_dir"] + "/" + model_name
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

tensorboard_logs_dir = ml_config["tensorboard_logs_dir"] + "/" + model_name
if not os.path.exists(tensorboard_logs_dir):
    os.makedirs(tensorboard_logs_dir)

optimizer = getattr(tensorflow.keras.optimizers, ml_config["optimizer"])
optimizer = optimizer(learning_rate=ml_config["learning_rate"])

# Load data
train_x, train_y = format_data(np.load(train_output_path), np.load(train_features_path), n_samples=n_train_samples,
                                  timesteps_y=timesteps_y, timesteps_x=timesteps_x)
print("Train data: ", "x ->", train_x.shape, "y ->", train_y.shape)

test_x, test_y = format_data(np.load(test_output_path), np.load(test_features_path), n_samples=n_test_samples,
                             timesteps_y=timesteps_y, timesteps_x=timesteps_x)
print("Test data: ", "x ->", test_x.shape, "y ->", test_y.shape)

# construct the set of callbacks to log training process
callbacks = [EpochCheckpoint(weights_dir, every=ml_config["checkpoint_epoch"]),
             TensorBoard(log_dir=tensorboard_logs_dir, update_freq="epoch")]

print("[INFO] Using following hyperparameters:")
print("    Optimizer : {}\n    LR:{}\n    Loss:{}\n    Batch size:{}\n    Epochs:{}".format(ml_config["optimizer"],
                                                                                            ml_config["learning_rate"],
                                                                                            ml_config["loss"],
                                                                                            ml_config["batch_size"],
                                                                                            ml_config["n_epochs"]))

# Compile and fit model
model.compile(loss=ml_config["loss"], optimizer=optimizer)
model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), batch_size=ml_config["batch_size"],
          epochs=ml_config["n_epochs"], verbose=2, callbacks=callbacks)

