from ray import tune
from monsterurl import get_monster

def train_agent_for_environment(EnvClass, env_config, experiment_name):
    analysis = tune.run(
        "DQN",
        stop={'training_iteration': 100},
        checkpoint_freq=10,
        keep_checkpoints_num=1,
        checkpoint_score_attr='episode_reward_mean',
        checkpoint_at_end=1,
        config = {
            "env": EnvClass,
            "lr": tune.grid_search([1e-2, 1e-4, 1e-6]),
            "env_config": env_config
        }
    )

    best_trial = analysis.get_best_trial(metric='episode_reward_mean', mode='max')
    best_checkpoint = analysis.get_best_checkpoint(
        metric='episode_reward_mean', trial=best_trial,  mode='max'
    )

    file_name = f'results/{experiment_name}_{get_monster()}.txt'

    with open(file_name, 'w') as f:
        print(env_config['reward_function'], file=f)
        print(experiment_name, file=f)
        print(best_checkpoint, file=f)

    print(f'Wrote to {file_name}')

