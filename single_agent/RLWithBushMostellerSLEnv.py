"""
SL = 'sum reward, last round state'

The two letters before 'Env' denote reward and state

Reward:
S = sum of Bush-Mosteller contributions
P = proportion of agents that contributed > 0.5

State:
W = whole game visible
L = last round visible
"""

from RLWithBushMostellerEnv import RLWithBushMostellerEnv
from BushMostellerConstants import N, tmax

import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print

import gym
import numpy as np

class RLWithBushMostellerSLEnv(RLWithBushMostellerEnv):
    def __init__(self, config):
        super().__init__(config)
        # give last round and current round number
        self.observation_space = gym.spaces.Tuple((gym.spaces.Box(low=0, high=1, shape=(1, N + 1), dtype=np.float32), gym.spaces.Discrete(tmax + 1)))

    def get_reward(self):
        if self.current_round < tmax:
            return 0
        else:
            # get sum of all agents except the RL agent
            return sum(sum(at[:N]) for at in self.all_at)

    def get_state(self):
        return ([self.all_at[self.current_round - 1]], self.current_round)

if __name__ == '__main__':
    # all_rewards = []
    # num_iters = 1
    # for _ in range(num_iters):
    #     env = RLWithBushMostellerSLEnv({})

    #     for i in range(tmax):
    #         state, reward, done, info = env.step(100)
    #         print(state)
    #         print(reward)
    #         if i == tmax - 1:
    #             all_rewards.append(reward)

    # print(sum(all_rewards) / num_iters)

    env_config = {
        "env": RLWithBushMostellerSLEnv,
        "lr": 1e-4,
        "num_workers": 1,
        "env_config": {}
    }

    ray.shutdown()
    ray.init()
    trainer = DQNTrainer(env=RLWithBushMostellerSLEnv, config=env_config)

    NUM_TRAINING_ITERATIONS = 5000

    episode_reward_means = open('sl_results/episode_reward_means.txt', 'w+')
    full_results = open('sl_results/full_results.txt', 'w+')
    checkpoint_paths = open('sl_results/checkpoints.txt', 'w+')

    for i in range(NUM_TRAINING_ITERATIONS):
        if i % 100 == 0:
            checkpoint = trainer.save()
            print(checkpoint)
            checkpoint_paths.write(str(i) + ": ")
            checkpoint_paths.write(checkpoint)
            checkpoint_paths.write('\n')

        result = trainer.train()
        episode_reward_mean = result.get('episode_reward_mean')
        print(str(episode_reward_mean))
        episode_reward_means.write(str(episode_reward_mean))
        episode_reward_means.write('\n')
        full_results.write(pretty_print(result))
        full_results.write('\n')

    episode_reward_means.close()
    full_results.close()
    checkpoint_paths.close()

    total_reward = 0
    env = RLWithBushMostellerSLEnv({})
    state = env.reset()

    done = False
    while not done:
        action = trainer.compute_action(state)
        state, reward, done, info = env.step(action)
        # print(state)
        total_reward += reward

    print("Total reward = " + str(total_reward))