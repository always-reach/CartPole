import dataclasses
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation


@dataclasses.dataclass
class KerasModel:
    input_dim: int
    output_dim: int
    model: Sequential

    def __init__(self):
        self.input_dim = 4
        self.output_dim = 2
        self.model = self.create_model()
        self.model.compile(loss="mse", optimizer="adam", metrics=["mae", "mse"])

    def create_model(self):
        model = Sequential()
        model.add(Dense(256, activation="relu", input_dim=self.input_dim))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(self.output_dim, activation="sigmoid"))
        return model

    def q_values(self, status):
        return self.model.predict(np.asarray([status]))[0]

    def learn(self, x, q, batch_size):
        self.model.fit(np.asarray(x), np.asarray(q), verbose=0, batch_size=batch_size)
