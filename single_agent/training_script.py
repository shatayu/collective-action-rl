import numpy as np
import gym
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from gym.spaces import Discrete, Box, MultiDiscrete

import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print

import os

NUM_TRAINING_ITERATIONS = 5000

def train_agent_for_environment(train_env, test_env, name, env_config):
    ray.shutdown()
    ray.init()
    trainer = DQNTrainer(env=train_env, config=env_config)

    if not os.path.exists(f'{name}'):
        os.makedirs(f'{name}')

    episode_reward_means = open(f'{name}/episode_reward_means.txt', 'w+')
    full_results = open(f'{name}/full_results.txt', 'w+')
    checkpoint_paths = open(f'{name}/checkpoints.txt', 'w+')

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
    env = test_env
    state = env.reset()

    done = False
    while not done:
        action = trainer.compute_action(state)
        state, reward, done, _ = env.step(action)
        # print(state)
        total_reward += reward

    print("Total reward = " + str(total_reward))