import unittest
from snake.snake import Game, Snake
# import numpy as np


class TestGame(unittest.TestCase):
    def setUp(self):
        self.g = Game()

    def testAvailableActions(self):
        all_directions = {(0, 1), (0, -1), (-1, 0), (1, 0)}
        test_pos = {
            (0, 1): [(0, 1), (0, 2)],
            (0, -1): [(0, 2), (0, 1)],
            (-1, 0): [(2, 0), (1, 0)],
            (1, 0): [(1, 0), (2, 0)]
        }
        for direction, pos in test_pos.items():
            self.pos = pos
            self.assertEqual(self.g.snake.available_directions, all_directions - set([direction]))


if __name__ == '__main__':
    unittest.main()
