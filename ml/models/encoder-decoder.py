from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, RepeatVector, Dropout
from tensorflow.keras import Sequential
import json

data_config = json.load(open("data_config.json"))
timesteps_x = data_config["input_timesteps"]
n_features = len(data_config["input_features"])

# Model definition
model = Sequential()
model.add(LSTM(64, activation="relu", input_shape=(timesteps_x, n_features)))
model.add(RepeatVector(1))
model.add(LSTM(64, activation="relu", return_sequences=True))
model.add(TimeDistributed(Dense(32)))
model.add(TimeDistributed(Dense(data_config["output_timesteps"])))
