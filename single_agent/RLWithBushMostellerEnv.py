# -*- coding: utf-8 -*-
"""Bush-Mosteller With RL.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Uw9VeVUAVgjuqTCGN_ywuqCFQINHA-PU

We will use the Bush-Mosteller approach:
![probability for cooperating for public goods game](https://journals.plos.org/ploscompbiol/article/file?type=thumbnail&id=info:doi/10.1371/journal.pcbi.1005034.e021)

![the stimulus (s) is based on whether or not the player’s actual payoff met some aspiration (expectation)](https://journals.plos.org/ploscompbiol/article/file?type=thumbnail&id=info:doi/10.1371/journal.pcbi.1005034.e002)



* basic idea behind this is that whether or not a person contributes is based on whether or not they think the group’s work met an expectation of theirs
* the initial condition p<sub>1</sub> is from the uniform density on [0, 1] independently for different players.
* First, we define p<sub>t</sub> as the expected contribution that the player makes in round t. We draw the actual contribution at from the truncated Gaussian distribution whose mean and standard deviation are equal to pt and 0.2, respectively. If a<sub>t</sub> falls outside the interval [0, 1], we discard it and redraw at until it falls within [0, 1]. Second, we introduce a threshold contribution value X, distinct from A, used for regarding the action to be either cooperative or defective.
    * X = 0.3 and 0.4 for conditional cooperative behavior
    * beta = 0.4, A = 0.9
"""

import numpy as np
import gym
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from gym.spaces import Discrete, Box, MultiDiscrete

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search
import math

from BushMostellerConstants import iteNum, N, tmax, a, std, beta, A, X

tf = try_import_tf()

class RLWithBushMostellerEnv(gym.Env):
    def __init__(self, config):
        self.aveCont = [0.0] * tmax
        self.net = self.completeNet()
        self.pt = [0.0] * N
        self.At = [0.0] * N
        self.st = [0.0] * N
        self.at = [0.0] * (N + 1)
        self.payoff = [0.0] * N
        self.current_round = 0
        self.all_at = [([0.0] * (N + 1)) for _ in range(tmax)]

        self.action_space = gym.spaces.Discrete(101)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(tmax, N + 1), dtype=np.float32)

        self.initialize(self.net, self.payoff, self.at, self.pt, self.st, self.At)

    def initialize(self, net, payoff, at, pt, st, At):
        for i in range(N):
            payoff[i] = 0
            pt[i] = np.random.rand()
            at[i] = np.random.normal() * std + pt[i] # initial contribution

            while at[i] < 0 or at[i] > 1:
                at[i] = np.random.normal() * std + pt[i] # discard irrational contribution
            
            At[i] = A
            st[i] = 0

        self.all_at[self.current_round] = at.copy()

    def reset(self):
        self.aveCont = [0.0] * tmax
        self.net = self.completeNet()
        self.pt = [0.0] * N
        self.At = [0.0] * N
        self.st = [0.0] * N
        self.at = [0.0] * (N + 1)
        self.payoff = [0.0] * N
        self.current_round = 0
        self.all_at = [([0.0] * (N + 1)) for _ in range(tmax)]

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,1), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(tmax, N), dtype=np.float32)

        self.initialize(self.net, self.payoff, self.at, self.pt, self.st, self.At)
        return self.get_state()
        
    def completeNet(self):
        net = [[] for i in range(N + 1)]

        for i in range(N + 1):
            for j in range(N + 1):
                if i != j:
                    net[i].append(j)
        
        return net

    def updatePGG(self, net, payoff, at, pt, st, At):
        # compute payoffs
        for i in range(len(payoff)):
            pool = 0
            for j in range(len(net[i])): # collect contributions from neighbors
                pool += at[net[i][j]] * a
            
            pool += at[i] * a
            payoff[i] = 1 - at[i] + pool / (len(net[i]) + 1)
        
        # update pt[]
        for i in range(N):
            st[i] = math.tanh(beta * (payoff[i] - At[i]))

            if (at[i] >= X): # cooperation
                if st[i] >= 0:
                    pt[i] = pt[i] + (1 - pt[i]) * st[i]
                else:
                    pt[i] = pt[i] + pt[i] * st[i]
            else: # defected
                if st[i] >= 0:
                    pt[i] = pt[i] - pt[i] * st[i]
                else:
                    pt[i] = pt[i] - (1 - pt[i]) * st[i]
        
        # draw contribution for next round
        for i in range(N):
            at[i] = np.random.normal() * std + pt[i]
            while at[i] < 0 or at[i] > 1:
                at[i] = np.random.normal() * std + pt[i]

        self.all_at[self.current_round] = at.copy()
    
    def step(self, action_input):
        action = action_input / 100.0
        if self.current_round == 0:
            self.at[N] = action
            self.all_at[0][N] = action
        elif self.current_round < tmax:
            self.at[N] = action
            self.updatePGG(self.net, self.payoff, self.at, self.pt, self.st, self.At)
            self.aveCont[self.current_round] += np.mean(self.at)

        self.current_round += 1
        
        return self.get_state(), self.get_reward(), self.current_round >= tmax, {}
    
    def get_state(self):
        # return self.all_at
        pass

    def get_reward(self):
        # return 0.0 if self.current_round < tmax else sum(sum(at) for at in self.all_at)
        pass