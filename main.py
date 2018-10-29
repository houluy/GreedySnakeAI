# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:25:20 2018

@author: houlu
"""

import pygame
import sys
import random
import itertools


class Shape:
    def __init__(self, background, base, block_size, shape=pygame.draw.rect):
        self.background = background
        self.base = base
        self.pos = []
        self.block_size = block_size
        self.size = (self.block_size, self.block_size)
        self.shape = shape

    def draw(self):
        for p in self.pos:
            p = self._c2p(p)
            rect = (*p, *self.size)
            self.shape(self.background, self.color, rect, 0)

    def _c2p(self, pos):
        return self.base[0] + pos[0]*self.size[0], self.base[1] + pos[1]*self.size[1]


class Rect(Shape):
    pass


class Circle(Shape):
    def __init__(self, **kwargs):
        super().__init__(shape=pygame.draw.circle, **kwargs)

    def draw(self):
        for p in self.pos:
            p = self._c2p(p)
            self.shape(self.background, self.color, p, self.block_size // 2)

    def _c2p(self, pos):
        base = super()._c2p(pos)
        return base[0] + self.block_size // 2, base[1] + self.block_size // 2


class Frame(Rect):
    def __init__(self, **kwargs):
        self.color = kwargs.pop('color')
        super().__init__(**kwargs)
        self.pos = [(0, 0)]


class Snake(Rect):
    ' Direction: UP: (0, -1) DOWN: (0, 1) RIGHT: (1, 0) LEFT: (-1, 0)'

    UP = (0, -1)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.length = 3
        self.pos = [(2, 0), (1, 0), (0, 0)]
        self._body_color = (249, 96, 96)
        self._head_color = (5, 156, 75)

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


class Food(Circle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = (52, 115, 243)

    @staticmethod
    def _gen(allowed):
        return random.choice(allowed)

    def replenish(self, allowed):
        self.pos = [self._gen(allowed)]


class Game:

    def __init__(self, expansion=1):
        pygame.init()
        self.background_size = int(320 * expansion)
        self.frame_size = int(300 * expansion)
        self.block_size = int(15 * expansion)
        self.number = (self.background_size - self.frame_size) // self.block_size
        self.grid = set(itertools.product(range(5), range(5)))
        self.background_color = (183, 222, 232)

        self.background_shape = (self.background_size, self.background_size)
        self.frame_shape = (self.frame_size, self.frame_size)
        self.frame_base = tuple((self.background_size - self.frame_size) // 2 for _ in range(2))
        self.frame_color = (219, 238, 244)

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
        self.frame.draw()

        self.snake = Snake(background=self.background, base=self.frame_base, block_size=self.block_size)
        self.snake.draw()

        self.food = Food(background=self.background, base=self.frame_base, block_size=self.block_size)
        self.food.replenish(self.allowed)
        self.food.draw()

    @property
    def allowed(self):
        return list(self.grid - self.snake.occupation)

    def run(self):
        # Event loop
        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            self.screen.blit(self.background, (0, 0))
            pygame.display.flip()

    def __getattr__(self, name):
        return getattr(self.snake, name)


g = Game(3)
g.run()
pygame.quit()
sys.exit(0)