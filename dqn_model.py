import dataclasses
import numpy as np
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint


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
    model_file_path: str

    def __init__(self):
        self.input_dim = 4
        self.output_dim = 2
        self.alpha = 0.3
        self.gamma = 0.9
        self.replay_memory_size = 400
        self.replay_memory = []
        self.minibatch_size = 128
        self.epsilon = 1
        self.update_epsilon_variable = 0.999
        self.model_file_path = "./weights.hdf5"
        self.model = self.create_model()
        self.model.compile(loss="mse", optimizer=Adam(), metrics=["mse"])

    def create_model(self) -> Sequential:
        model = Sequential()
        model.add(Dense(32, activation="relu", input_dim=self.input_dim))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.output_dim, activation="sigmoid"))
        return model

    def load_weights(self):
        self.model.load_weights(self.model_file_path)

    def fit_call_back(self) -> list:
        li_cb = [ModelCheckpoint(self.model_file_path, monitor="loss", verbose=1,
                                 save_best_only=True, save_weights_only=True)]
        return li_cb

    def epsilon_greedy(self, state):
        if np.random.uniform(0, 1) > self.epsilon:
            q_value = self.q_values(state)
            print(q_value)
            return np.argmax(q_value)
        else:
            return np.random.choice([0, 1])

    def update_epsilon(self):
        if self.epsilon > 0.1:
            self.epsilon *= self.update_epsilon_variable

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
                state_t = experience.get("state_t")
                state_t1 = experience.get("state_t1")
                reward = experience.get("reward")

                state_list.append(state_t)

                q_table = self.q_values(state_t)
                expected_q = self.q_values(state_t1)
                q_table[action] += self.alpha * (reward + self.gamma * np.max(expected_q) - q_table[action])

                q_list.append(q_table)
            self.model.fit(np.asarray(state_list), np.asarray(q_list), verbose=0, batch_size=self.minibatch_size,
                           callbacks=self.fit_call_back())
