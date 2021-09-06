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
from tabulate import tabulate

tf = try_import_tf()

iteNum = 1000
N = 3 # number of agents EXCLUDING the RL agent
tmax = 25 # number of rounds

a = 1.6 # multiply factor for PGG

std = 0.2 # std of distribution from which contribution is drawn
beta = 0.4
A = 1.0
X = 0.5 # cooperativeness criteria

class Env(gym.Env):
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

        self.all_at[self.current_round] = at
        print('Initializing self.all_at[{0}]'.format(self.current_round))

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
        return self.all_at
        
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
        
        self.all_at[self.current_round] = at
        print('Setting self.all_at[{0}]'.format(self.current_round))
    
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

        return self.all_at, self.get_reward(), self.current_round >= tmax, {}

    def get_reward(self):
        return 0.0 if self.current_round < tmax else sum(sum(at) for at in self.all_at)

env_config = {
    "env": Env,
    "lr": 1e-4,
    "num_workers": 1,
    "env_config": {}
}

from ray import tune
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print

ray.shutdown()
ray.init()

model_location = '/home/shatayu/ray_results/DQN_Env_2021-08-27_01-30-41p41_rv5v/checkpoint_5100/checkpoint-5100'

test_agent = DQNTrainer(env=Env, config={})
test_agent.restore(model_location)

NUM_TRIALS = 1

total_game = [([0.0] * (N + 1)) for _ in range(tmax)]
total_reward = 0

for _ in range(NUM_TRIALS):
    total_reward = 0

    env = Env({})
    state = env.reset()

    done = False
    final_state = None

    while not done:
        action = test_agent.compute_action(state)
        state, reward, done, info = env.step(action)
        print(tabulate(state, headers=["Agent " + str(i) for i in range(N + 1)]))
        print()
        total_reward += reward
        final_state = state
    
    for current_round in range(len(final_state)):
        for agent in range(len(final_state[current_round])):
            total_game[current_round][agent] += final_state[current_round][agent]

for current_round in range(len(final_state)):
        for agent in range(len(final_state[current_round])):
            total_game[current_round][agent] /= NUM_TRIALS

total_reward /= NUM_TRIALS

print(tabulate(total_game, headers=["Agent " + str(i) for i in range(N + 1)]))
print('Average reward = ' + str(total_reward))