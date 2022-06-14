import dataclasses
import gym
from gym import wrappers, Env


@dataclasses.dataclass
class GymEnv:
    env: Env = gym.make("CartPole-v0")
