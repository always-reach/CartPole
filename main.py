import numpy as np
import logging
from gym_env import GymEnv
from dqn_model import DQNModel
from matplotlib import pyplot

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def train():
    gym_env: GymEnv = GymEnv()
    dqn_model: DQNModel = DQNModel()
    num_episodes: int = 3000
    reward_list = []
    for episode in range(num_episodes):
        print("episode", episode)
        state_t = gym_env.env.reset()
        game = False
        gross_reward = 0
        while not game:
            action = dqn_model.epsilon_greedy(state_t)
            state_t1, reward, game, info = gym_env.env.step(action)
            gross_reward += reward
            replay_dict: dict = dict(state_t=state_t, state_t1=state_t1, action=action, reward=reward)
            dqn_model.replay_memory.append(replay_dict)
            state_t = state_t1

            gym_env.env.render()
        reward_list.append(gross_reward)
        dqn_model.experience_replay()
        dqn_model.update_epsilon()

    gym_env.env.close()
    pyplot.plot(reward_list)
    pyplot.show()


def test():
    gym_env: GymEnv = GymEnv()
    dqn_model: DQNModel = DQNModel()
    dqn_model.epsilon = 0
    dqn_model.load_weights()
    while True:
        state = gym_env.env.reset()
        game = False
        while not game:
            action = dqn_model.epsilon_greedy(state)
            state, reward, game, info = gym_env.env.step(action)
            gym_env.env.render()


if __name__ == '__main__':
    train()
# test()
