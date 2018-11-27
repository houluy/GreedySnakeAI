import time
import pygame
from shapes import Frame
import numpy as np


class Window:
    def __init__(self, number, block_size, speed=0.5, expansion=2):
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

        self.bcolor = (84, 255, 159)
        self.fcolor = (46, 139, 87)

        self.screen = pygame.display.set_mode(self.background_shape)
        pygame.display.set_caption('Greedy Snake AI')
        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill(self.background_color)

        self.colors = {

        }

        self.frame = Frame(color=self.frame_color, block_size=self.frame_size)

        self.directions = dict(zip([pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN], range(4)))

        self.brushes = {
            'rect': pygame.draw.rect,
            'circle': pygame.draw.circle,
        }

    def draw(self, game):
        state = game.state
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()
        for ind, val in np.ndenumerate(state):
            fc, bc, shape = self.colors[val]
            co =
            self.brush(shape, bc, ind)
            self.brush(shape, fc, ind)
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

    def brush(self, shape, color, co, width=0):
        brush = self.brushes[shape]
        brush(self.background, color, co, width)

    def __del__(self):
        pygame.quit()