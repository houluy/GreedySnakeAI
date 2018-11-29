# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:25:20 2018

@author: houlu
"""
import sys
import random
import itertools
import numpy as np
from collections.abc import Sequence


class Snake(Sequence):

    UP = (-1, 0)
    DOWN = (1, 0)
    RIGHT = (0, 1)
    LEFT = (0, -1)
    INITIAL_POS = [(1, 4), (1, 3), (1, 2), (1, 1)]

    def __init__(self, number):
        self.number = number
        self.all_directions = [self.LEFT, self.RIGHT, self.UP, self.DOWN]
        self.action_to_direction = dict(enumerate(self.all_directions))

    def __getitem__(self, item):
        return self.pos[item]

    def __len__(self):
        return len(self.pos)

    def __iter__(self):
        return iter(self.pos)

    def __contains__(self, item):
        return item in self.pos

    @staticmethod
    def _tuple_add(t1, t2):
        return t1[0] + t2[0], t1[1] + t2[1]

    @property
    def occupation(self):
        return set(self)

    @property
    def inversion(self):
        return -self.direction[0], -self.direction[1]

    def move(self):
        self.pos.insert(0, self._tuple_add(self.head, self.direction))
        self.pos.pop(-1)
        self.head = self[0]

    def turn(self, action_index):
        if action_index is None:
            return
        try:
            next_direction = self.action_to_direction[action_index]
        except KeyError:
            return
        if next_direction == self.inversion:
            return
        else:
            self.direction = next_direction

    def eat(self, food):
        if self.head == food:
            self.pos.append(self.pos[-1])
            return True
        else:
            return False

    @property
    def available_directions(self):
        return [ind
                for ind, action in enumerate(self.all_directions)
                if not (self._tuple_add(self.head, action) == self[1])
        ]

    def reset(self):
        self.pos = self.INITIAL_POS[:]
        self.head = self[0]
        self.direction = self.RIGHT


class Food:

    def __init__(self):
        self.color = (52, 115, 243)

    @staticmethod
    def _gen(allowed):
        return random.choice(allowed)

    def replenish(self, allowed):
        self.pos = self._gen(allowed)


class Game:

    def __init__(self, number):
        self.number = number
        self.state_number = self.number + 2
        self.boundry = (1, self.number + 1)
        self.grid = set(itertools.product(range(*self.boundry), range(*self.boundry)))
        self.value = {
            'head': 1,
            'body': -1,
            'food': 2,
            'wall': -2,
            'earth': 0
        }

        self._state = np.zeros((self.state_number, self.state_number))
        self.snake = Snake(number=self.number)
        self.food = Food()

        self.reset()

        self.score = 0

    @property
    def allowed(self):
        return list(self.grid - self.snake.occupation)

    @property
    def actions(self):
        return self.snake.available_directions

    def control(self): pass

    @property
    def state(self):
        self._state[1:self.number + 1, 1:self.number + 1] = self.value['earth']
        self._state[self.food.pos[0], self.food.pos[1]] = self.value['food']
        for body in self.snake:
            self._state[body[0], body[1]] = self.value['body']
        self._state[self.snake.head[0], self.snake.head[1]] = self.value['head']
        # return np.array([self._state]).reshape(self.state_number, self.state_number, 1)
        return self._state

    @property
    def eat(self):
        return self.snake.eat(self.food.pos)

    @property
    def death(self):
        body = self.snake[1:]
        if self.snake.head in body:
            return True
        if (not self.boundry[0] <= self.snake.head[0] < self.boundry[1])\
                or (not self.boundry[0] <= self.snake.head[1] < self.boundry[1]):
            return True
        return False

    @staticmethod
    def action(neural):
        return neural.action

    def interact(self, action_index):
        self.snake.turn(action_index)
        self.snake.move()

    def reset(self):
        self.snake.reset()
        self.food.replenish(self.allowed)
        self._state[[0, self.number + 1], :] = self.value['wall']
        self._state[:, [0, self.number + 1]] = self.value['wall']

    def new_food(self):
        self.score += 1
        self.food.replenish(self.allowed)

    @property
    def instant_reward(self):
        if self.eat:
            return 10
        elif self.death:
            return -10
        else:
            return 0

    def play(self, engine=None):
        if engine:
            engine.draw(self.state)
        while True:
            if engine:
                action_index = engine.action
                if action_index is False:
                    return
            else:
                action_index = self.snake.direction
            self.snake.turn(action_index)
            self.snake.move()
            if self.eat:
                self.new_food()
            if engine:
                engine.draw(self.state)
            if self.death:
                return


if __name__ == '__main__':
    base_size = 20
    expansion = 1.5
    number = 10
    g = Game(number=number)
    from window import Window
    window = Window(number=number, block_size=base_size, expansion=expansion, speed=0.1)
    g.play(engine=window)
    sys.exit(0)
