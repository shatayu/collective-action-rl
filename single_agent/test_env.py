from Env import Env
from random import randint

e = Env({})

def print_state(state):
    for _round in state:
        displayRound = [round(100 * contribution) / 100.0 for contribution in _round]
        print(displayRound)
    print('-**************')

# tmax - length of game - is 25
for _ in range(5):
    contribution = randint(0, 10) / 10.0
    state, reward, done, info = e.step(contribution)
    print_state(state)