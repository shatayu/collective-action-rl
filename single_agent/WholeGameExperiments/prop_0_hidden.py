from RLWithBushMostellerWholeGameEnv import RLWithBushMostellerWholeGameEnv

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training_script import train_agent_for_environment

REWARD_FUNCTION = 'proportion'
NUM_ROUNDS_HIDDEN = 0

env_config = {
    'reward_function': REWARD_FUNCTION,
    'num_rounds_hidden': NUM_ROUNDS_HIDDEN
}

experiment_name = f'{REWARD_FUNCTION}_{NUM_ROUNDS_HIDDEN}_hidden'

train_agent_for_environment(RLWithBushMostellerWholeGameEnv, env_config, experiment_name)
