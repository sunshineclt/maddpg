from gym import spaces
import numpy as np


class ToyEnv:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.observation_space = [spaces.Box(low=-1, high=-1, shape=(1, ))] * num_agents
        self.action_space = [spaces.Box(low=-1, high=1, shape=(1, ))] * num_agents
        self.step = 0
        self.n = num_agents

    def get_reward(self, action):
        action = np.clip(action, -1, 1)
        if np.all(action > 0):
            reward = [np.sum(action)] * self.num_agents
        else:
            reward = 0
            for act in action:
                if act[0] < 0:
                    reward -= act[0] / 2
            reward = [reward] * self.num_agents
        return reward

    def step(self, action):
        self.step += 1
        reward = self.get_reward(action)
        return [np.ones(shape=(1, ))] * self.num_agents, reward, [self.step >= 1] * self.num_agents

    def reset(self):
        self.step = 0
        return [np.ones(shape=(1, ))] * self.num_agents
