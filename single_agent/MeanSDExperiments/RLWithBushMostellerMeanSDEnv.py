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

import numpy as np

import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
import gym

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RLWithBushMostellerEnv import RLWithBushMostellerEnv
from BushMostellerConstants import N, tmax

class RLWithBushMostellerMeanSDEnv(RLWithBushMostellerEnv):
    def __init__(self, *args):
        super().__init__(*args)
        self.observation_space = gym.spaces.Box(low=-1, high=tmax + 1, shape=(tmax * 2 + 1, ), dtype=np.float32)

    def get_state(self):
        game = np.array(self.all_at)
        game = np.delete(game, N - 1, axis=1)

        means = game.mean(axis=1)
        standard_deviations = game.std(axis=1)

        means[:self.num_rounds_hidden] = -1
        standard_deviations[:self.num_rounds_hidden] = -1

        return np.concatenate([means, standard_deviations, np.array([self.current_round])]).astype(np.float32)