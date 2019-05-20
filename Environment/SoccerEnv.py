import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from dm_control.rl.environment import StepType
from gym import spaces


class SoccerEnv:
    def __init__(self, team_size, time_limit):
        """
        Create a object that bridges Soccre env from dm_control to Ray MultiAgentEnv interface
        :param team_size: how many agents are in one team
        :param time_limit: timestep of the env (1second = 40timestep)
        """
        self.team_size = team_size
        self.agent_num = 2 * team_size
        self.n = self.agent_num
        self.env = dm_soccer.load(team_size=team_size, time_limit=time_limit)
        # action space creation
        self.action_space = [spaces.Box(-1.0, 1.0, shape=[3])] * self.agent_num
        # observation space creation
        self.feature_space_length = 0
        for feature_name, feature_spec in self.env.observation_spec()[0].items():
            self.feature_space_length += np.prod(feature_spec.shape)
        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(self.feature_space_length, ))] * self.n
        self.info = {}

    def reset(self):
        """
        Reset the environment
        :return: the initial observation
        """
        timestep = self.env.reset()
        return self.extract_observation(timestep)

    def step(self, action):
        """
        Make one timestep ahead in the env
        :param action: action for each agent
        :return: (observation, rewards, dones, infos)
        """
        timestep = self.env.step(self.transform_action(action))
        observations = self.extract_observation(timestep)
        rewards = self.extract_reward(timestep)
        dones = self._extract_done(timestep)
        infos = self._extract_info(timestep)
        return observations, rewards, dones, infos

    def transform_action(self, action):
        return action

    def extract_observation(self, timestep):
        observation_n = []
        for agent_index in range(self.agent_num):
            observation = []
            for feature_name in timestep.observation[agent_index].keys():
                if feature_name == "stats_teammate_spread_out":
                    # this is the only bool feature
                    observation.append(1 if timestep.observation[agent_index][feature_name] else 0)
                else:
                    # other values are all real value
                    observation.extend(timestep.observation[agent_index][feature_name].flatten().tolist())
            observation_n.append(np.array(observation))
        return observation_n

    def extract_reward(self, timestep):
        if timestep.step_type == StepType.FIRST:
            rewards = [0.0] * self.agent_num
            return rewards
        blue_running_rewards = 0.0
        red_running_rewards = 0.0
        for agent_index in range(self.team_size):
            blue_running_rewards += timestep.observation[agent_index]["stats_vel_to_ball"]
            red_running_rewards += timestep.observation[self.team_size + agent_index]["stats_vel_to_ball"]
        self.info["blue_running_rewards"] = blue_running_rewards
        self.info["red_running_rewards"] = red_running_rewards

        rewards = [float(float(timestep.reward[agent_index]) +
                         blue_running_rewards * 0.1 +
                         timestep.observation[agent_index]["stats_vel_ball_to_goal"] * 1 +
                         (timestep.observation[agent_index]["stats_home_score"] == 1) * 1000)
                   for agent_index in range(self.team_size)]
        rewards.extend([float(float(timestep.reward[self.team_size + agent_index]) +
                              red_running_rewards * 0.1 +
                              timestep.observation[self.team_size + agent_index]["stats_vel_ball_to_goal"] * 1 +
                              (timestep.observation[self.team_size + agent_index]["stats_away_score"] == 1) * 1000)
                       for agent_index in range(self.team_size)])
        if timestep.observation[0]["stats_home_score"] == 1:
            print("Blue got a goal! ")
            print("rewards: ", rewards)
        elif timestep.observation[0]["stats_away_score"] == 1:
            print("Red got a goal! ")
            print("rewards: ", rewards)
        return rewards

    def _extract_done(self, timestep):
        done = timestep.step_type == StepType.LAST
        dones = [done] * self.agent_num
        return dones

    def _extract_info(self, timestep):
        # infos = {"blue_agent_" + str(agent_index): {
        #     "running_reward": self.info["blue_running_rewards"]
        # }
        #     for agent_index in range(self.team_size)}
        # infos.update({"red_agent_" + str(agent_index): {
        #     "running_reward": self.info["red_running_rewards"]
        # }
        #     for agent_index in range(self.team_size)})
        infos = {}
        return infos

    def close(self):
        self.env.close()

    def __del__(self):
        self.close()
