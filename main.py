from snake.algorithm import DQN
from snake.snake import Game
from snake.window import Window

base_size = 20
expansion = 1.5
number = 5
window = Window(number=number, block_size=base_size, expansion=expansion, speed=0.2)
g = Game(number=number, window=window)
dqn = DQN(game=g)
#dqn.train(None)
g.play(dqn)
