import json

from keras.layers import Dense, LSTM, Lambda
from keras.models import Sequential

from ml.layers.attention_decoder import AttentionDecoder

data_config = json.load(open("data_config.json"))
timesteps_x = data_config["input_timesteps"]
timesteps_y = data_config["output_timesteps"]
n_features = len(data_config["input_features"])

# Model definition
model = Sequential()
model.add(LSTM(150, activation="relu", input_shape=(timesteps_x, n_features), return_sequences=True))
model.add(AttentionDecoder(150, output_dim=n_features))
model.add(Lambda(lambda x: x[:, 0:1, :]))
model.add(Dense(1))
model.summary()
