import numpy as np
import logging
from gym_env import GymEnv
from keras_models import KerasModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run():
    gym_env: GymEnv = GymEnv()
    keras_model: KerasModel = KerasModel()
    num_episodes: int = 3000
    max_score = 200.0
    score = []
    for episode in range(num_episodes):
        print("episode", episode)
        state = gym_env.env.reset()
        gym_env.env.render()
        x, q, a, r = [], [], [], []
        game = False
        while not game:
            q_value = keras_model.q_values(state)
            action = np.argmax(q_value)
            x.append(state)
            a.append(action)
            q.append(q_value)

            state, reward, game, info = gym_env.env.step(action)
            r.append(reward)

        alpha = 0.1
        batch_size = len(x)
        gross_reward = np.sum(r)

        for index in range(batch_size):
            past_action = a[index]
            q[index][past_action] = (1 - alpha) * q[index][past_action] + alpha * (gross_reward / max_score)
        keras_model.learn(x, q, batch_size)
        score.append(gross_reward)

    logger.info("Average Reward: %.3f" % np.mean(score))
    gym_env.env.close()


if __name__ == '__main__':
    run()
