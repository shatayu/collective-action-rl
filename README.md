# collective-action-rl

Install this repository with `git clone https://github.com/shatayu/collective-action-rl.git` or some other mechanism.

## Legend:

In summary:

* PL = agent tried to maximize proportion of others' contributions over 0.5 with knowledge of the last round only
* PW = agent tried to maximize proportion of others' contributions over 0.5 with knowledge of the whole game
* SL = agent tried to maximize sum of others' contributions with knowledge of the last round only
* SW = agent tried to maximize sum of others' contribution with knowledge of the whole game

### Explanation of the legend

These experiments tested two things: varying the objectives of the agents (proportion of contributions over 0.5 and total sum of contributions) and varying the information they had available (last round only and entire game). As a result, a shorthand was developed to identify these experiments.

P = proportion of contributions over 0.5
S = sum of contributions

L = last round known to RL agent
W = whole game known to RL agent

The two letters put together tell you what the agent is trying to do and what information it knows.

## How to Run the Code

These experiments were ran using Python 3.7.9. Other versions of Python 3 may work, but 3.7.9 definitely works. Run `python --version` to see which version of Python is active on your machine.

The documents below use the PL experiment as an example. For PW, SL, and SW, just replace "PL" with your experiment.

* To run an experiment, run `python RLWithBushMostellerPLEnv.py`.
    * These will generate files called `episode_reward_means.txt`, `full_results.txt`, and `checkpoints.txt` in the folder `pl_results`.
* To plot episode means **after running scripts for all 4 experiments**, run `python plot_episode_reward_means.py`
    * If there is an `Error: <some folder> does not exist`, then run the corresponding experiment.
