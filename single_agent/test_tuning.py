import ray
from ray import tune
from ray.rllib.agents.dqn import ApexTrainer, DEFAULT_CONFIG

from RLWithBushMostellerEnv import RLWithBushMostellerEnv
from BushMostellerConstants import N, tmax

import gym
import numpy as np

class RLWithBushMostellerPLEnv(RLWithBushMostellerEnv):
    def __init__(self, config):
        super().__init__(config)
        # give last round and current round number
        self.observation_space = gym.spaces.Tuple((gym.spaces.Box(low=0, high=1, shape=(1, N + 1), dtype=np.float32), gym.spaces.Discrete(tmax + 1)))

    def get_reward(self):
        if self.current_round < tmax:
            return 0
        else:
            # get number of contributions from BM agents that were above 0.5
            return sum([len(list(filter(lambda x: x > 0.5, at[:3]))) for at in self.all_at])

    def get_state(self):
        return ([self.all_at[self.current_round - 1]], self.current_round)


analysis = tune.run(
    ApexTrainer,
    stop={'training_iteration': 2500},
    config={
        "env": RLWithBushMostellerPLEnv,
        "num_workers": 31,
        "num_gpus": 0,
        "lr": tune.grid_search([0.01, 0.001, 0.0001, 0.00001, 0.000001]),
        "gamma": tune.grid_search([0.99, 0.9, 0.5])
    },
    checkpoint_freq=100,
    checkpoint_at_end=True
)

trial = analysis.get_best_trial(metric='episode_reward_mean', mode='max')
checkpoint = analysis.get_best_checkpoint(trial=trial, mode='max')

final_checkpoint = open('test_results/episode_reward_means.txt', 'w+')
final_checkpoint.write(checkpoint)
final_checkpoint.close()




