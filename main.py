import numpy as np
import logging
from gym_env import GymEnv
from dqn_model import DQNModel
from matplotlib import pyplot

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run():
    gym_env: GymEnv = GymEnv()
    dqn_model: DQNModel = DQNModel()
    num_episodes: int = 3000
    reward_list = []
    for episode in range(num_episodes):
        print("episode", episode)
        state = gym_env.env.reset()
        game = False
        gross_reward = 0
        while not game:
            q_value: list[float] = dqn_model.q_values(state)
            action = dqn_model.epsilon_greedy(state)
            state, reward, game, info = gym_env.env.step(action)
            gross_reward += reward
            replay_dict: dict = dict(state=state, action=action, q_table=q_value, reward=reward)
            dqn_model.replay_memory.append(replay_dict)

            gym_env.env.render()
        reward_list.append(gross_reward)
        dqn_model.experience_replay()
        dqn_model.update_epsilon()

    gym_env.env.close()
    pyplot.plot(reward_list)
    pyplot.show()


if __name__ == '__main__':
    run()
