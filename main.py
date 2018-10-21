# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:25:20 2018

@author: houlu
"""

import pygame
import sys
import random


class Shape:
    def draw(self, background): pass


class Snake(Shape):
    '''
    Direction:
        UP: (0, -1)
        DOWN: (0, 1)
        RIGHT: (1, 0)
        LEFT: (-1, 0)
    '''

    UP = (0, -1)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)

    def __init__(self):
        self.length = 3
        self.pos = [(2, 0), (1, 0), (0, 0)]
        self._body_color = (255, 255, 255)
        self._head_color = (255, 0, 0)
        self._size = (20, 20)
        self._dir = self.UP # Direction

    def block(self, pos, background, head=False):
        rect = (*pos, *self._size)
        color = self._head_color if head else self._body_color
        pygame.draw.rect(background, color, rect, 0)

    def draw(self, background):
        for p in self.pos:
            head = True if p == self.pos[0] else False
            self.block(self._c2p(p), background, head=head)

    def _c2p(self, pos):
        return pos[0]*self._size[0], pos[1]*self._size[1]


class Food(Shape):
    def __init__(self):
        self._color = (255, 0, 0)

    @staticmethod
    def _gen(allowed):
        return random.choice(allowed)


class Game:
    BACKGROUD_COLOR = (0, 0, 0) # Black

    def __init__(self):
        pygame.init()
        self.snake = Snake()
        self.screen = pygame.display.set_mode((200, 200))
        pygame.display.set_caption('Greedy Snake AI')
        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill(self.BACKGROUD_COLOR)
        self.snake.draw(self.background)

    def run(self):
        # Event loop
        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            self.screen.blit(self.background, (0, 0))
            pygame.display.flip()


g = Game()
g.run()
pygame.quit()
sys.exit(0)