# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:25:20 2018

@author: houlu
"""

import pygame
import sys
import random
import itertools
import time
import numpy as np
import collections.abc


class Shape:

    def __init__(self, block_size, shape=pygame.draw.rect):
        self.pos = []
        self.shape = shape
        self.block_size = block_size

    def draw(self, window):
        if not isinstance(self.pos, list):
            pos = [self.pos]
        else:
            pos = self.pos
        for p in pos:
            p = self._c2p(p, window)
            rect = (*p, self.block_size, self.block_size)
            self.shape(window.background, self.color, rect, 0)

    def _c2p(self, pos, window):
        return window.base[0] + pos[0]*self.block_size, window.base[1] + pos[1]*self.block_size

    @staticmethod
    def _tuple_add(t1, t2):
        return t1[0] + t2[0], t1[1] + t2[1]


class Rect(Shape):
    pass


class Circle(Shape):

    def __init__(self, **kwargs):
        super().__init__(shape=pygame.draw.circle, **kwargs)

    def draw(self, window):
        if not isinstance(self.pos, list):
            pos = [self.pos]
        else:
            pos = self.pos
        for p in pos:
            p = self._c2p(p, window)
            self.shape(window.background, self.color, p, self.block_size // 2)

    def _c2p(self, pos, window):
        base = super()._c2p(pos, window)
        return base[0] + self.block_size // 2, base[1] + self.block_size // 2


class Frame(Rect):

    def __init__(self, **kwargs):
        self.color = kwargs.pop('color')
        super().__init__(**kwargs)
        self.pos = (0, 0)


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
        self._body_color = (249, 96, 96)
        self._head_color = (5, 156, 75)
        self.all_directions = [self.LEFT, self.RIGHT, self.UP, self.DOWN]
        self.action_to_direction = dict(enumerate(self.all_directions))

    def __getitem__(self, item):
        return self.pos[item]

    def block(self, pos, window, head=False):
        rect = (*pos, self.block_size, self.block_size)
        color = self._head_color if head else self._body_color
        pygame.draw.rect(window.background, color, rect, 0)

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
        self.grid = set(itertools.product(range(self.number), range(self.number)))
        self.value = {
            'head': 1,
            'body': -1,
            'food': 2,
            'wall': -2
        }

        self._state = np.zeros((self.number + 2, self.number + 2))
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
        return self._state

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
                self.score += 1
                self.food.replenish(self.allowed)
            if self.death:
                return


class Window:
    def __init__(self, number, block_size, speed=0.1, expansion=2):
        pygame.init()
        self.speed = speed
        self.number = number
        self.block_size = int(block_size * expansion)
        self.frame_size = int(self.block_size*self.number)
        self.background_size = int(self.frame_size + 20*expansion)
        self.background_color = (183, 222, 232)

        self.background_shape = (self.background_size, self.background_size)
        self.frame_shape = (self.frame_size, self.frame_size)
        self.base = tuple((self.background_size - self.frame_size) // 2 for _ in range(2))
        self.frame_color = (219, 238, 244)

        self.screen = pygame.display.set_mode(self.background_shape)
        pygame.display.set_caption('Greedy Snake AI')
        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill(self.background_color)

        self.frame = Frame(color=self.frame_color, block_size=self.frame_size)

        self.directions = dict(zip([pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN], range(4)))

    def draw(self, *args):
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()
        self.frame.draw(self)
        for obj in args:
            obj.draw(self)
        time.sleep(self.speed)

    @property
    def action(self):
        if pygame.event.peek(pygame.QUIT):
            return False
        elif pygame.event.peek(pygame.KEYDOWN):
            events = pygame.event.get(pygame.KEYDOWN)
            current_event = events[-1]
            return self.directions[current_event.key]
        else:
            return None


if __name__ == '__main__':
    expansion = 1.5
    block_size = int(15 * expansion)
    g = Game(number=20, block_size=block_size)
    window = Window(number=20, block_size=block_size, expansion=expansion, speed=0.5)
    g.play(engine=window)
    pygame.quit()
    sys.exit(0)
