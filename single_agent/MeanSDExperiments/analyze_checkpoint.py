from ray.rllib.agents.dqn import DQNTrainer
from RLWithBushMostellerMeanSDEnv import RLWithBushMostellerMeanSDEnv

import sys
import numpy as np
import matplotlib.pyplot as plt
import ray

NUM_SAMPLES = 50
NUM_GAMES_IN_SAMPLE = 50

reward_function = None
experiment_name = None
checkpoint = None

with open(f'results/{sys.argv[1]}', 'r') as cf:
    reward_function = cf.readline().rstrip('\n')
    experiment_name = cf.readline().rstrip('\n')
    checkpoint = cf.readline().rstrip('\n')

env_config = {
    "reward_function": reward_function,
    "num_rounds_hidden": 0
}

ray.shutdown()
ray.init()

trainer = DQNTrainer({
    "env": RLWithBushMostellerMeanSDEnv,
    "env_config": env_config
})

trainer.load_checkpoint(checkpoint)

all_samples_rewards = []
for sample_index in range(NUM_SAMPLES):
    print(f'Evaluating sample #{sample_index + 1}')
    total_rewards = []
    for game_index in range(NUM_GAMES_IN_SAMPLE):
        total_reward = 0
        env = RLWithBushMostellerMeanSDEnv(env_config)
        state = env.reset()

        done = False
        while not done:
            action = trainer.compute_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    all_samples_rewards.append(total_rewards)

plt.hist(all_samples_rewards, bins='sqrt')
plt.ylabel(f'Reward (mean = {np.format_float_positional(np.mean(all_samples_rewards), precision=3)}, sd = {np.format_float_positional(np.std(all_samples_rewards), precision=3)})')
plt.title(f'Episode Reward Means for {experiment_name}')
plt.savefig(f'results/{sys.argv[1]}_episode_reward_means_chart.png')