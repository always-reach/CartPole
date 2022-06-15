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
    score = []
    for episode in range(num_episodes):
        print("episode", episode)
        state = gym_env.env.reset()
        x, q, a, r = [], [], [], []
        game = False
        while not game:
            q_value: list[float] = keras_model.q_values(state)
            action: int = np.argmax(q_value)
            state, reward, game, info = gym_env.env.step(action)

            x.append(state)
            a.append(action)
            q.append(q_value)

            gym_env.env.render()
            r.append(reward)

        batch_size = len(x)
        # 1エピソードをバッチサイズとする
        q_tabel = keras_model.update_q(q, x, r, a, batch_size)

        gross_reward = np.sum(r)

        keras_model.model.fit(np.asarray(x), np.asarray(q_tabel), verbose=0, batch_size=batch_size)
        score.append(gross_reward)

    logger.info("Average Reward: %.3f" % np.mean(score))
    gym_env.env.close()


if __name__ == '__main__':
    run()
