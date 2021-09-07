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

from RLWithBushMostellerEnv import RLWithBushMostellerEnv
from BushMostellerConstants import N, tmax

def print_state(state):
    for _round in state:
        displayRound = [round(100 * contribution) / 100.0 for contribution in _round]
        print(displayRound)
    print('**************')

class RLWithBushMostellerSLEnv(RLWithBushMostellerEnv):
    def get_reward(self):
        if self.current_round < tmax:
            return 0
        else:
            # get sum of all agents except the RL agent
            return sum(sum(at[:N]) for at in self.all_at)

    def get_state(self):
        return self.all_at

if __name__ == '__main__':
    all_rewards = []
    num_iters = 1
    for _ in range(num_iters):
        env = RLWithBushMostellerSLEnv({})

        for i in range(tmax):
            state, reward, done, info = env.step(100)
            if i == tmax - 1:
                print_state(state)
                all_rewards.append(reward)

    print(sum(all_rewards) / num_iters)