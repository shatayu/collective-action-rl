# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

folders = ['sl_results', 'pl_results', 'sw_results', 'pw_results']

for folder in folders:
    f = open(folder + '/episode_reward_means.txt', 'r')
    lines = f.readlines()
    means = [float(line) for line in lines]

    plt.plot(means)

    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Episode Reward Means for DQN Agent (' + folder + ')')
    plt.savefig(folder + '/episode_reward_means_chart.png')

    f.close()