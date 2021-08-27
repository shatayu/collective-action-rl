
import numpy as np
import gym
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from gym.spaces import Discrete, Box, MultiDiscrete
from ray.rllib import MultiAgentEnv

class CollectiveActionEnv(MultiAgentEnv):
    def __init__(self, return_agent_actions = False, part=False):
        # constants
        self.NUM_TURNS = 5
        self.NUM_AGENTS = 17
        self.MAX_CONTRIBUTION = 10

        # in game stuff
        self.current_turn = 0
        self.total_pot = 0
        self.rewards = [0 for i in range(self.NUM_AGENTS)]
        self.board = np.ndarray(shape=(self.NUM_AGENTS, self.NUM_TURNS))
        for i in range(self.NUM_AGENTS):
            for j in range(self.NUM_TURNS):
                self.board[i][j] = -1

        # environment
        self.observation_space = gym.spaces.Box(low=-self.NUM_AGENTS - 1, high=self.MAX_CONTRIBUTION * self.NUM_AGENTS * self.NUM_TURNS + 1, shape=(1,))
        self.action_space = gym.spaces.Discrete(self.MAX_CONTRIBUTION + 1)
        self.agent_ids = list(range(self.NUM_AGENTS))

    def reset(self):
        self.current_turn = 0
        self.total_pot = 0
        self.rewards = [0 for i in range(self.NUM_AGENTS)]

        for i in range(self.NUM_AGENTS):
            for j in range(self.NUM_TURNS):
                self.board[i][j] = -1

        return self.find_observations()

    def find_observations(self):
        contribution_sum = 0
        for i in range(self.NUM_AGENTS):
            contribution_sum += self.board[i][self.current_turn]
        values = [np.array([contribution_sum]) for i in range(self.NUM_AGENTS)]
        
        return self._make_dict(values)

    def play_game(self, action_dict):
        for i in range(self.NUM_AGENTS):
            all_contributions = [j for j in range(self.MAX_CONTRIBUTION + 1)]
            chosen_action = np.random.choice(a=all_contributions, size=None, replace=True, p=action_dict[i])

            self.board[i][self.current_turn] = float(chosen_action)
            self.total_pot += chosen_action
        
    def find_rewards(self, action_dict):
        # compute everyone's individual rewards
        for i in range(self.NUM_AGENTS):
            chosen_action = self.board[i][self.current_turn]

            amount_saved = self.MAX_CONTRIBUTION - chosen_action
            self.rewards[i] += float(amount_saved)
                
        if self.current_turn == self.NUM_TURNS - 1:
            for i in range(self.NUM_AGENTS):
                self.rewards[i] += 1.6 * self.total_pot / float(self.NUM_AGENTS)
        
        # since our agent is trying to maximize the group good its reward will be the group good
        # total_reward = sum(self.rewards)
        # total_reward_array = [total_reward] * self.NUM_AGENTS

        return self._make_dict(self.rewards)
    
    def find_dones(self):
        done = self.current_turn == self.NUM_TURNS
        values = [done for i in range(self.NUM_AGENTS)]
        dones = self._make_dict(values)
        dones["__all__"] = done

        return dones
    
    def find_pool(self):
        temp_board = np.copy(self.board)
        for i in range(len(temp_board)):
            for j in range(len(temp_board[0])):
                if temp_board[i][j] == -1:
                    temp_board[i][j] = 0

        return np.sum(temp_board)
    
    def find_reward(self, agent_id):
        # get action from prior round
        assert(self.current_turn > 0)
        prior_round_action = self.board[agent_id][self.current_turn - 1]

        return self.MAX_CONTRIBUTION - prior_round_action + self.find_pool() / float(NUM_AGENTS)
    
    def find_stimulus(self, agent_id):
        reward = self.find_reward(agent_id) / 10.0
        beta = 0.4
        A = 0.9
        return np.tanh(beta * (reward - A))

    def step(self, action_dict):
        self.play_game(action_dict)
        obs, rew, info = self.find_observations(), self.find_rewards(action_dict), {}
        self.current_turn += 1
        done = self.find_dones()

        return obs, rew, done, info

    def _make_dict(self, values):
        return dict(zip(self.agent_ids, values))

from ray.tune.registry import register_env

def env_creator(_):
    return CollectiveActionEnv()

single_env = CollectiveActionEnv()
env_name = "CollectiveActionEnv"
register_env(env_name, env_creator)

obs_space = single_env.observation_space
act_space = single_env.action_space
NUM_AGENTS = single_env.NUM_AGENTS

def gen_policy(i):
    return (None, obs_space, act_space, {'agent_id': i})

policy_graphs = {}

for i in range(NUM_AGENTS):
    policy_graphs['agent-' + str(i)] = gen_policy(i)

def policy_mapping_fn(agent_id):
        return 'agent-' + str(agent_id)

config = {
    "learning_starts": 4000,
    "multiagent": {
        "policies": policy_graphs,
        "policy_mapping_fn": policy_mapping_fn
    },
    "env": "CollectiveActionEnv"
}

from ray.rllib.contrib.maddpg import MADDPGTrainer
from tabulate import tabulate

from ray.tune.logger import pretty_print
import json

ray.init(num_cpus=None)

saved_competitive_trainer = MADDPGTrainer(config=dict(config, **{
        "env": CollectiveActionEnv,
    }), env=CollectiveActionEnv)

saved_competitive_trainer.restore("/home/shatayu/ray_results/MADDPG_CollectiveActionEnv_2021-04-22_14-44-02qeqzrkag/checkpoint_50000/checkpoint-50000")

saved_cooperative_trainer = MADDPGTrainer(config=dict(config, **{
        "env": CollectiveActionEnv,
    }), env=CollectiveActionEnv)

saved_cooperative_trainer.restore("/home/shatayu/ray_results/MADDPG_CollectiveActionEnv_2021-04-22_14-43-161kean90v/checkpoint_50000/checkpoint-50000")



def play_one_game(competitive_trainer, cooperative_trainer):
    env = CollectiveActionEnv()
    states = env.reset()
    all_done = False

    action_grid = []

    while not all_done:
        action = {}
        if env.current_turn == 0:
            for agent_id, _ in states.items():
                # randomly choose first action
                possible_contributions = list(range(env.MAX_CONTRIBUTION + 1))
                random_first_action = np.random.choice(possible_contributions)
                action[agent_id] = [0.0 for i in range(env.MAX_CONTRIBUTION + 1)]
                action[agent_id][random_first_action] = 1.0 # set the probability of the first action being chosen to 1.0
        else:
            for agent_id, agent_state in states.items():
                # use algorithm to choose action
                policy_id = config['multiagent']['policy_mapping_fn'](agent_id)

                stimulus = env.find_stimulus(agent_id)
                if stimulus >= 0:
                    trainer = cooperative_trainer
                else:
                    trainer = competitive_trainer
                action[agent_id] = trainer.compute_action(agent_state, policy_id=policy_id)

        states, rewards, done, info = env.step(action)
        all_done = done['__all__']

    # print(tabulate(env.board, headers=["Agent " + str(i) for i in range(env.NUM_AGENTS)]))
    # print(env.board.shape)
    # print(env.board[0]) # is this an agent's contributions or round 0's? 
    # print("-----------")
    # print(tabulate(env.board, headers=["Agent " + str(i) for i in range(env.NUM_AGENTS)]))
    # print(tabulate(env.board.transpose(), headers=["Agent " + str(i) for i in range(env.NUM_AGENTS)]))

    return env.board


game = play_one_game(saved_competitive_trainer, saved_cooperative_trainer)
print(tabulate(game.transpose(), headers=["Agent " + str(i) for i in range(17)]))

def play_n_games(N, competitive_trainer, cooperative_trainer):
    return [play_one_game(competitive_trainer, cooperative_trainer) for i in range(N)]


def average_contributions_each_round(n_games_array):
    num_agents = len(n_games_array[0])
    num_turns = len(n_games_array[0][0])

    average_contributions = np.zeros((num_agents, num_turns))

    for agent in range(num_agents):
        for turn in range(num_turns):
            average_contributions[agent][turn] = np.mean(n_games_array[game][agent][turn] for game in range(len(n_games_array)))
    
    return average_contributions

n_games_results = play_n_games(1000, saved_competitive_trainer, saved_cooperative_trainer)
average_contributions = average_contributions_each_round(n_games_results)
print(average_contributions.shape)

import matplotlib.pyplot as plt

for i in range(len(average_contributions)):
    plt.plot(average_contributions[i])

plt.xlabel('Round')
plt.ylabel('Sum Contribution')
plt.title('Sum Contributions of Each Agent in Each Round (1000 Games, Sum State)')
# plt.savefig('charts/sum_state_sum_contributions.png')

print("Average contribution across all rounds: ")
print(np.mean(average_contributions, axis=1))
print("Standard deviation contribution across all rounds: ")
print(np.std(average_contributions, axis=1))

ray.shutdown()