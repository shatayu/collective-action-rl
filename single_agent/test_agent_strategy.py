import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from RLWithBushMostellerEnv import RLWithBushMostellerEnv
from tabulate import tabulate

env_config = {
    "env": RLWithBushMostellerEnv,
    "lr": 1e-4,
    "num_workers": 1,
    "env_config": {}
}

ray.shutdown()
ray.init()

model_location = '/home/shatayu/ray_results/DQN_Env_2021-08-27_01-30-41p41_rv5v/checkpoint_5100/checkpoint-5100'

test_agent = DQNTrainer(env=RLWithBushMostellerEnv, config={})
test_agent.restore(model_location)

NUM_TRIALS = 1

N = 3
tmax = 25

total_game = [([0.0] * (N + 1)) for _ in range(tmax)]
total_reward = 0

for _ in range(NUM_TRIALS):
    total_reward = 0

    env = RLWithBushMostellerEnv({})
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