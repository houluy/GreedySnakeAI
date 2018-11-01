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


class Shape:

    def __init__(self, background, base, block_size, shape=pygame.draw.rect):
        self.background = background
        self.base = base
        self.pos = []
        self.block_size = block_size
        self.size = (self.block_size, self.block_size)
        self.shape = shape

    def draw(self):
        if not isinstance(self.pos, list):
            pos = [self.pos]
        else:
            pos = self.pos
        for p in pos:
            p = self._c2p(p)
            rect = (*p, *self.size)
            self.shape(self.background, self.color, rect, 0)

    def _c2p(self, pos):
        return self.base[0] + pos[0]*self.size[0], self.base[1] + pos[1]*self.size[1]

    @staticmethod
    def _tuple_add(t1, t2):
        return t1[0] + t2[0], t1[1] + t2[1]


class Rect(Shape):
    pass


class Circle(Shape):

    def __init__(self, **kwargs):
        super().__init__(shape=pygame.draw.circle, **kwargs)

    def draw(self):
        if not isinstance(self.pos, list):
            pos = [self.pos]
        else:
            pos = self.pos
        for p in pos:
            p = self._c2p(p)
            self.shape(self.background, self.color, p, self.block_size // 2)

    def _c2p(self, pos):
        base = super()._c2p(pos)
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

    def __init__(self, **kwargs):
        self.number = kwargs.pop('wall')
        super().__init__(**kwargs)
        self.pos = [(4, 0), (3, 0), (2, 0), (1, 0), (0, 0)]
        self._body_color = (249, 96, 96)
        self._head_color = (5, 156, 75)
        self.direction = self.RIGHT
        self.head = self.pos[0]
        self.all_directions = {
            pygame.K_LEFT: self.LEFT,
            pygame.K_RIGHT: self.RIGHT,
            pygame.K_UP: self.UP,
            pygame.K_DOWN: self.DOWN,
        }

    def block(self, pos, head=False):
        rect = (*pos, *self.size)
        color = self._head_color if head else self._body_color
        pygame.draw.rect(self.background, color, rect, 0)

    def draw(self):
        for p in self.pos:
            head = True if p == self.pos[0] else False
            self.block(self._c2p(p), head=head)

    @property
    def occupation(self):
        return set(self.pos)

    @property
    def inversion(self):
        return -self.direction[0], -self.direction[1]

    def move(self):
        self.pos.insert(0, self._tuple_add(self.head, self.direction))
        self.pos.pop(-1)
        self.head = self.pos[0]

    def turn(self, event):
        try:
            next_direction = self.all_directions[event.key]
        except KeyError:  # Other keys are pressed
            return
        if next_direction == self.inversion:
            return
        else:
            self.direction = next_direction

    @property
    def death(self):
        body = self.pos[1:]
        if self.head in body:
            return True
        if self.head[0] < 0 or self.head[0] > self.number - 1 or self.head[1] < 0 or self.head[1] > self.number - 1:
            return True
        return False

    def eat(self, food):
        if self.head == food:
            self.pos.append(self.pos[-1])
            return True
        else:
            return False


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

    def __init__(self, speed=0.1, expansion=2):
        pygame.init()
        self.speed = speed
        self.background_size = int(320 * expansion)
        self.frame_size = int(300 * expansion)
        self.block_size = int(15 * expansion)
        self.number = self.frame_size // self.block_size
        self.grid = set(itertools.product(range(self.number), range(self.number)))
        self.background_color = (183, 222, 232)

        self.background_shape = (self.background_size, self.background_size)
        self.frame_shape = (self.frame_size, self.frame_size)
        self.frame_base = tuple((self.background_size - self.frame_size) // 2 for _ in range(2))
        self.frame_color = (219, 238, 244)

        self.value = {
            'head': 1,
            'body': -1,
            'food': 2,
            'wall': -2
        }

        self._state = np.zeros((self.number + 1, self.number + 1))
        self._state[[0, self.number], :] = self.value['wall']
        self._state[:, [0, self.number]] = self.value['wall']

        self.screen = pygame.display.set_mode(self.background_shape)
        pygame.display.set_caption('Greedy Snake AI')
        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill(self.background_color)

        self.frame = Frame(
            background=self.background,
            base=self.frame_base,
            block_size=self.frame_size,
            color=self.frame_color
        )

        self.snake = Snake(
            background=self.background,
            base=self.frame_base,
            block_size=self.block_size,
            wall=self.number
        )

        self.food = Food(
            background=self.background,
            base=self.frame_base,
            block_size=self.block_size
        )

        self.food.replenish(self.allowed)
        self._draw()

        self.score = 0

    def _draw(self):
        self.frame.draw()
        self.snake.draw()
        self.food.draw()

    @property
    def allowed(self):
        return list(self.grid - self.snake.occupation)

    def control(self): pass

    def run(self):
        while 1:
            if pygame.event.peek(pygame.QUIT):
                return
            elif pygame.event.peek(pygame.KEYDOWN):
                events = pygame.event.get(pygame.KEYDOWN)
                current_event = events[-1]
                self.snake.turn(current_event)
            self.screen.blit(self.background, (0, 0))
            pygame.display.flip()
            self.snake.move()
            if self.eat:
                self.score += 1
                self.food.replenish(self.allowed)
            self._draw()
            if self.death:
                return
            time.sleep(self.speed)

    @property
    def state(self):
        self._state[1:self.number - 1, 1:self.number - 1] = 0
        self._state[self.food.pos[1] + 1, self.food.pos[0] + 1] = self.value['food']
        for body in self.snake.pos:
            self._state[body[1] + 1, body[0] + 1] = self.value['body']
        self._state[self.snake.head[1] + 1, self.snake.head[0] + 1] = self.value['head']
        return self._state

    @property
    def eat(self):
        return self.snake.eat(self.food.pos[0])

    @property
    def death(self):
        return self.snake.death


g = Game(speed=1)
g.run()
pygame.quit()
sys.exit(0)

