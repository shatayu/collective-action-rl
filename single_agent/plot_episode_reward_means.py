# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

f = open('current_episode_reward_means.txt', 'r')
lines = f.readlines()
means = [float(line) for line in lines]

plt.plot(means)

plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.title('Episode Reward Means for DQN Agent')
plt.savefig('episode_reward_means_chart.png')