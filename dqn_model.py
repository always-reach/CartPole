import dataclasses
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense


@dataclasses.dataclass
class DQNModel:
    input_dim: int
    output_dim: int
    model: Sequential
    alpha: float
    gamma: float
    replay_memory_size: int
    replay_memory: list[dict]
    minibatch_size: int
    epsilon: float

    def __init__(self):
        self.input_dim = 4
        self.output_dim = 2
        self.alpha = 0.2
        self.gamma = 0.99
        self.replay_memory_size = 400
        self.replay_memory = []
        self.minibatch_size = 128
        self.epsilon = 0.002
        self.model = self.create_model()
        self.model.compile(loss="mse", optimizer="adam", metrics=["mae", "mse"])

    def create_model(self) -> Sequential:
        model = Sequential()
        model.add(Dense(256, activation="relu", input_dim=self.input_dim))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(self.output_dim, activation="sigmoid"))
        return model

    def epsilon_greedy(self, state):
        if np.random.uniform(0, 1) > self.epsilon:
            q_value = self.q_values(state)
            return np.argmax(q_value)
        else:
            return np.random.choice([0, 1])

    def q_values(self, status) -> list[float]:
        return self.model.predict(np.asarray([status]))[0]

    def experience_replay(self) -> None:
        if len(self.replay_memory) > self.replay_memory_size:
            minibatch_indexes = np.random.randint(0, len(self.replay_memory), self.minibatch_size)
            state_list = []
            q_list = []
            for index in minibatch_indexes:
                experience: dict = self.replay_memory[index]
                action = experience.get("action")
                q_table = experience.get("q_table")
                state = experience.get("state")
                reward = experience.get("reward")

                state_list.append(state)

                expected_q = self.q_values(state)
                q_table[action] += self.alpha * (reward + self.gamma * np.max(expected_q) - q_table[action])
                q_list.append(q_table)
            self.model.fit(np.asarray(state_list), np.asarray(q_list), verbose=0, batch_size=self.minibatch_size)
