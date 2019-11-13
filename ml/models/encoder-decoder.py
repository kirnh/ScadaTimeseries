from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, RepeatVector, Dropout
from tensorflow.keras import Sequential
import json

data_config = json.load(open("data_config.json"))
timesteps_x = data_config["input_timesteps"]
n_features = len(data_config["input_features"])

# Model definition
model = Sequential()
model.add(LSTM(64, activation="relu", input_shape=(timesteps_x, n_features), return_sequences=True))
model.add(LSTM(32, activation="relu"))
# model.add(Dropout(0.5))
model.add(RepeatVector(1))
model.add(LSTM(64, activation="relu", return_sequences=True))
# model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(32)))
model.add(TimeDistributed(Dense(1)))
