iteNum = 1000
N = 3 # number of agents EXCLUDING the RL agent
tmax = 25 # number of rounds

a = 1.6 # multiply factor for PGG

std = 0.2 # std of distribution from which contribution is drawn
beta = 0.4
A = 1.0
X = 0.4 # cooperativeness criteria for Bush-Mosteller algorithm
COOPERATIVE_CONSTANT_FOR_REWARD = 0.4 # cooperativeness criteria for reward function