import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from Env import Env

env_config = {
    "env": Env,
    "lr": 1e-4,
    "num_workers": 1,
    "env_config": {}
}

ray.shutdown()
ray.init()
trainer = DQNTrainer(env=Env, config=env_config)

NUM_TRAINING_ITERATIONS = 1

episode_reward_means = open('episode_reward_means.txt', 'a')
full_results = open('full_results.txt', 'a')
checkpoint_paths = open('checkpoints.txt', 'a')

for i in range(NUM_TRAINING_ITERATIONS):
    if i % 100 == 0:
        checkpoint = trainer.save()
        print(checkpoint)
        # checkpoint_paths.write(str(i) + ": ")
        # checkpoint_paths.write(checkpoint)
        # checkpoint_paths.write('\n')

    result = trainer.train()
    episode_reward_mean = result.get('episode_reward_mean')
    print(str(episode_reward_mean))
    # episode_reward_means.write(str(episode_reward_mean))
    # episode_reward_means.write('\n')
    # full_results.write(pretty_print(result))
    # full_results.write('\n')

episode_reward_means.close()
full_results.close()
checkpoint_paths.close()

total_reward = 0
env = Env({})
state = env.reset()

done = False
while not done:
    action = trainer.compute_action(state)
    state, reward, done, info = env.step(action)
    print(state)
    total_reward += reward

print("Total reward = " + str(total_reward))