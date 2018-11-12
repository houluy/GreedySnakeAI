from algorithm import DQN
from snake import Game

g = Game(number=10, block_size=15)
dqn = DQN(ipt_size=10, out_size=4)

dqn.train(g)

