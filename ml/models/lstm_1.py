from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Dropout
from tensorflow.keras import Sequential
import json

data_config = json.load(open("data_config.json"))
timesteps_x = data_config["input_timesteps"]
n_features = len(data_config["input_features"])

# Model definition
model = Sequential()
model.add(LSTM(64, activation="relu", input_shape=(timesteps_x, n_features), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(32, activation="relu", input_shape=(timesteps_x, n_features)))
model.add(Dropout(0.5))
model.add(Dense(1))
