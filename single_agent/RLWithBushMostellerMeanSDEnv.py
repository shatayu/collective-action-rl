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

import numpy as np

import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print

from training_script import train_agent_for_environment

class RLWithBushMostellerMeanSDEnv(RLWithBushMostellerEnv):
    def get_state(self):
        game = np.array(self.all_at)

        # 4 x 25

        means = game.mean(axis=1)
        standard_deviations = game.std(axis=1)

        print(means.shape)
        print(standard_deviations.shape)

        means[:self.num_rounds_hidden] = -1
        standard_deviations[:self.num_rounds_hidden] = -1

        return np.concatenate([means, standard_deviations, np.array([self.current_round])])