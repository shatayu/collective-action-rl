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

class RLWithBushMostellerWholeGameEnv(RLWithBushMostellerEnv):
    def get_state(self):
        game = np.concatenate([
            np.array(self.all_at).reshape(-1, 1),
            np.array([self.current_round]).reshape(1, 1)
        ])

        game[:((N + 1) * self.num_rounds_hidden)] = -1

        return game.reshape(1, -1)[0]

if __name__ == "__main__":
    e = RLWithBushMostellerWholeGameEnv({
        'reward_function': 'proportion',
        'num_rounds_hidden': 2
    })

    start_state = e.reset()

    print(start_state.shape)
    print(start_state)