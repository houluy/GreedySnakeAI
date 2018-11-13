from algorithm import DQN
from snake import Game

number = 10
actions = 4
state = 12

g = Game(number=number, block_size=15)
dqn = DQN(ipt_size=state, out_size=actions)

dqn.train(g)

