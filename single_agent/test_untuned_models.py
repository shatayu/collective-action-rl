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

import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print

from BushMostellerConstants import N, tmax

from RLWithBushMostellerPLEnv import RLWithBushMostellerPLEnv
from RLWithBushMostellerPWEnv import RLWithBushMostellerPWEnv
from RLWithBushMostellerSLEnv import RLWithBushMostellerSLEnv
from RLWithBushMostellerSWEnv import RLWithBushMostellerSWEnv

import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

NUM_ITERATIONS = 5000

def test_agent(agent, env, filename, title, reward_description):
    rewards = []
    for i in range(NUM_ITERATIONS):
        if i % 100 == 0:
            print(f'Iteration {i} / {NUM_ITERATIONS}')

        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.compute_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        
        rewards.append(total_reward)
    
    test = np.array(rewards)
    print(test.shape)
    # save chart
    plt.hist(np.array(rewards))
    plt.title(title)
    plt.xlabel(reward_description)
    plt.ylabel('Frequency')
    plt.savefig(f'histograms/{filename}.png')
    plt.clf()

    # save rewards
    with open(f'final_rewards/{filename}.txt', 'w') as filehandle:
        for reward in rewards:
            filehandle.write(f'{reward}\n')

ray.shutdown()
ray.init()
# PLEnv
env_config = {
    "env": RLWithBushMostellerPLEnv,
    "lr": 1e-4,
    "num_workers": 1,
    "env_config": {}
}

agent = DQNTrainer(env=RLWithBushMostellerPLEnv, config=env_config)
agent.restore('/home/shatayu/ray_results/DQN_RLWithBushMostellerPLEnv_2021-09-08_20-45-28cu5wnf4g/checkpoint_4900/checkpoint-4900')

test_agent(
    agent,
    RLWithBushMostellerPLEnv({}),
    'RLWithBushMostellerPLEnvResults',
    'Proportion of BM Contributions Over 0.5 With Last Round Visible Only',
    'Proportion of BM Contributions Over 0.5'
)

# PWEnv
env_config = {
    "env": RLWithBushMostellerPWEnv,
    "lr": 1e-4,
    "num_workers": 1,
    "env_config": {}
}

agent = DQNTrainer(env=RLWithBushMostellerPWEnv, config=env_config)
agent.restore('/home/shatayu/ray_results/DQN_RLWithBushMostellerPWEnv_2021-09-08_20-46-28sr91ow0n/checkpoint_4900/checkpoint-4900')

test_agent(
    agent,
    RLWithBushMostellerPWEnv({}),
    'RLWithBushMostellerPWEnvResults',
    'Proportion of BM Contributions Over 0.5 With Whole Game Visible',
    'Proportion of BM Contributions Over 0.5'
)

# SLEnv
env_config = {
    "env": RLWithBushMostellerSLEnv,
    "lr": 1e-4,
    "num_workers": 1,
    "env_config": {}
}

agent = DQNTrainer(env=RLWithBushMostellerSLEnv, config=env_config)
agent.restore('/home/shatayu/ray_results/DQN_RLWithBushMostellerSLEnv_2021-09-08_20-47-12ej6p2921/checkpoint_4900/checkpoint-4900')

test_agent(
    agent,
    RLWithBushMostellerSLEnv({}),
    'RLWithBushMostellerSLEnvResults',
    'Sum of All BM Contributions With Last Round Visible',
    'Sum of All BM Contributions'
)

# SWEnv
env_config = {
    "env": RLWithBushMostellerSWEnv,
    "lr": 1e-4,
    "num_workers": 1,
    "env_config": {}
}

agent = DQNTrainer(env=RLWithBushMostellerSWEnv, config=env_config)
agent.restore('/home/shatayu/ray_results/DQN_RLWithBushMostellerSWEnv_2021-09-08_20-48-22s3lv8izr/checkpoint_4900/checkpoint-4900')

test_agent(
    agent,
    RLWithBushMostellerSWEnv({}),
    'RLWithBushMostellerSWEnvResults',
    'Sum of All BM Contributions With Whole Game Visible',
    'Sum of All BM Contributions'
)

