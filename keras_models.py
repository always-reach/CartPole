import dataclasses
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense


@dataclasses.dataclass
class KerasModel:
    input_dim: int
    output_dim: int
    model: Sequential
    alpha: float
    gamma: float

    def __init__(self):
        self.input_dim = 4
        self.output_dim = 2
        self.alpha = 0.2
        self.gamma = 0.99
        self.model = self.create_model()
        self.model.compile(loss="mse", optimizer="adam", metrics=["mae", "mse"])

    def create_model(self) -> Sequential:
        model = Sequential()
        model.add(Dense(256, activation="relu", input_dim=self.input_dim))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(self.output_dim, activation="sigmoid"))
        return model

    def q_values(self, status) -> list[float]:
        return self.model.predict(np.asarray([status]))[0]

    def update_q(self, q_table: list[list[float]], next_state: list[list[float]], reword: list[float],
                 action: list[int], batch_size: int) -> list[list[float]]:
        for index in range(batch_size):
            action_index = action[index]
            expected_q = self.q_values(next_state[index])
            q_table[index][action_index] += self.alpha * (
                    reword[index] + self.gamma * np.max(expected_q) - q_table[index][action_index])
        return q_table
