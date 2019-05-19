import numpy as np
from Environment.ToyEnv import ToyEnv


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def evaluate(arglist, trainers, is_toy=False):
    if is_toy:
        env = ToyEnv(arglist.num_agents)
    else:
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
    obs_n = env.reset()
    episode_step = 0
    episode_rewards = [0.0]
    while True:
        # get action
        if is_toy:
            action_n = [agent.p_debug["p_values"](obs_n[idx][None])[0][0] for idx, agent in enumerate(trainers)]
        elif arglist.scenario == 'simple_reference':
            action_n = [agent.p_debug["p_values"](obs_n[idx][None])[0][0] for idx, agent in enumerate(trainers)]
        else:
            action_n = [agent.p_debug["p_values"](obs_n[idx][None])[0][0] for idx, agent in enumerate(trainers)]

        # environment step
        def transform_action_to_tuple(raw_action_n):
            return [(action[:2], action[2:]) for action in raw_action_n]

        if not is_toy and arglist.scenario == 'simple_reference':
            new_obs_n, rew_n, done_n, info_n = env.step(transform_action_to_tuple(action_n))
        else:
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)

        episode_step += 1
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)
        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew

        if done or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)

        # save model, display training output
        if terminal and (len(episode_rewards) == arglist.evaluate_episode):
            print("Evaluate end! Run {} episodes, mean episode reward: {}".format(
                  arglist.evaluate_episode,
                  np.mean(episode_rewards)))
            break
    return np.mean(episode_rewards)
