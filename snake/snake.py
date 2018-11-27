# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:25:20 2018

@author: houlu
"""
import sys
import random
import itertools
import numpy as np
from shapes import Rect, Circle


class Snake(Rect):

    UP = (0, -1)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)
    INITIAL_POS = [(4, 0), (3, 0), (2, 0), (1, 0), (0, 0)]

    def __init__(self, **kwargs):
        self.number = kwargs.pop('number')
        super().__init__(**kwargs)
        self.reset()
        self._body_bcolor = (84, 255, 159)
        self._body_fcolor = (46, 139, 87)
        self._head_bcolor = (0, 191, 255)
        self._head_fcolor = (0, 0, 205)
        self._innerratio = 0.1
        self._difference = self.block_size * self._innerratio
        self.all_directions = [self.LEFT, self.RIGHT, self.UP, self.DOWN]
        self.action_to_direction = dict(enumerate(self.all_directions))

    def __getitem__(self, item):
        return self.pos[item]

    def block(self, pos, window, head=False):
        brect = (*pos, self.block_size, self.block_size)
        frect = (pos[0] + self._difference,
                 pos[1] + self._difference,
                 self.block_size * (1 - 2 * self._innerratio),
                 self.block_size * (1 - 2 * self._innerratio)
                )
        fcolor, bcolor = (self._head_fcolor, self._head_bcolor) \
            if head else (self._body_fcolor, self._body_bcolor)
        window.brush("rect", bcolor, brect, 0)
        window.brush("rect", fcolor, frect, 0)

    def draw(self, window):
        for p in self:
            head = True if p == self[0] else False
            self.block(self._c2p(p, window), window, head=head)

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
        next_direction = self.action_to_direction[action_index]
        if next_direction == self.inversion:
            return
        else:
            self.direction = next_direction

    @property
    def death(self):
        body = self[1:]
        if self.head in body:
            return True
        if self.head[0] < 0 or self.head[0] > self.number - 1 or self.head[1] < 0 or self.head[1] > self.number - 1:
            return True
        return False

    def eat(self, food):
        if self.head == food:
            self.pos.append(self[-1])
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


class Food(Circle):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = (52, 115, 243)

    @staticmethod
    def _gen(allowed):
        return random.choice(allowed)

    def replenish(self, allowed):
        self.pos = self._gen(allowed)


class Game:

    def __init__(self, number, block_size):
        self.number = number
        self.state_number = self.number + 2
        self.grid = set(itertools.product(range(self.number), range(self.number)))
        self.value = {
            'head': 1,
            'body': -1,
            'food': 2,
            'wall': -2
        }

        self._state = np.zeros((self.state_number, self.state_number))
        self._state[[0, self.number], :] = self.value['wall']
        self._state[:, [0, self.number]] = self.value['wall']
        self.block_size = block_size
        self.snake = Snake(number=self.number, block_size=self.block_size)

        self.food = Food(block_size=self.block_size)

        self.food.replenish(self.allowed)

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
        self._state[1:self.number - 1, 1:self.number - 1] = 0
        self._state[self.food.pos[1] + 1, self.food.pos[0] + 1] = self.value['food']
        for body in self.snake:
            self._state[body[1] + 1, body[0] + 1] = self.value['body']
        self._state[self.snake.head[1] + 1, self.snake.head[0] + 1] = self.value['head']
        return np.array([self._state]).reshape(self.state_number, self.state_number, 1)

    @property
    def eat(self):
        return self.snake.eat(self.food.pos)

    @property
    def death(self):
        return self.snake.death

    @staticmethod
    def action(neural):
        return neural.action

    def interact(self, action_index):
        self.snake.turn(action_index)
        self.snake.move()

    def reset(self):
        self.snake.reset()
        self.food.replenish(self.allowed)
        self._state[[0, self.number], :] = self.value['wall']
        self._state[:, [0, self.number]] = self.value['wall']

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
        while True:
            if engine:
                engine.draw(self.snake, self.food)
                action_index = engine.action
                if action_index is False:
                    return
            else:
                action_index = self.snake.direction
            self.snake.turn(action_index)
            self.snake.move()
            if self.eat:
                self.new_food()
            if self.death:
                return


if __name__ == '__main__':
    expansion = 2
    block_size = int(15 * expansion)
    g = Game(number=20, block_size=block_size)
    from window import Window
    window = Window(number=20, block_size=block_size, expansion=expansion, speed=0.5)
    g.play(engine=window)
    sys.exit(0)
