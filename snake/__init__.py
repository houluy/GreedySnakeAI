from algorithm import DQN
from snake import Game
from window import Window

number = 5
actions = 4
state = number + 2
block_size = 30

g = Game(number=number, block_size=block_size)
window = Window(number=number, block_size=block_size)
dqn = DQN(ipt_size=state, out_size=actions)

dqn.train(g, window)
