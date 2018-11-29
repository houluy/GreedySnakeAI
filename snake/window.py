import time
import pygame
import numpy as np


class Window:
    def __init__(self, number, block_size, speed=0.5, expansion=2):
        pygame.init()
        self.speed = speed
        self.number = number
        self.block_size = int(block_size * expansion)
        self.background_size = (int(self.number + 4)*self.block_size)
        self.background_color = (183, 222, 232)

        self.background_shape = (self.background_size, self.background_size)
        self.base = (self.block_size, self.block_size)

        self.hcolor = (0, 0, 128)  # body color

        self.ecolor = (135, 206, 255)  # empty color

        self.bcolor = (0, 100, 0)  # head color

        self.wcolor = (72, 61, 139)  # wall color

        self.fcolor = (178, 34, 34)  # food color

        self.inner = 0.1
        self.iblock = int(self.block_size * (1 - 2 * self.inner))

        self.screen = pygame.display.set_mode(self.background_shape)
        pygame.display.set_caption('Greedy Snake AI')
        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill(self.background_color)

        self.outwidth = self.block_size // 20

        self.colors = {
            0: (self.ecolor, 0, "rect"),
            1: (self.hcolor, self.outwidth, "rect"),  # head
            -1: (self.bcolor, self.outwidth, "rect"),  # body
            2: (self.fcolor, self.outwidth, "rect"),  # food
            -2: (self.wcolor, self.outwidth, "rect"),  # wall
        }

        self.directions = dict(zip([pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN], range(4)))

        self.brushes = {
            'rect': pygame.draw.rect,
            'circle': pygame.draw.circle,
        }

    def _c2p(self, pos, shift=0):
        shift = self.block_size * shift
        return self.base[1] + pos[1]*self.block_size + shift, self.base[0] + pos[0]*self.block_size + shift

    def _c2c(self, pos):
        return self.base[1] + pos[1] * self.block_size + self.block_size // 2,\
               self.base[0] + pos[0] * self.block_size + self.block_size // 2

    def draw(self, state):
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()
        for ind, val in np.ndenumerate(state):
            color, width, shape = self.colors[val]
            # if shape == 'rect':
            co = (*self._c2p(ind), self.block_size, self.block_size)
            sco = (*self._c2p(ind, shift=self.inner), self.iblock, self.iblock)
            pygame.draw.rect(self.background, color, co, width)
            pygame.draw.rect(self.background, color, sco, 0)
            # elif shape == 'circle':
            #     center, radius = self._c2c(ind), self.block_size // 2
            #     scenter, sradius = self._c2c(ind), int(self.iblock // 2)
            #     pygame.draw.circle(self.background, color, center, radius, width)
            #     pygame.draw.circle(self.background, color, scenter, sradius, 0)

        time.sleep(self.speed)

    @property
    def action(self):
        if pygame.event.peek(pygame.QUIT):
            return False
        elif pygame.event.peek(pygame.KEYDOWN):
            events = pygame.event.get(pygame.KEYDOWN)
            current_event = events[-1]
            try:
                d = self.directions[current_event.key]
            except KeyError:
                return None
            else:
                return d
        else:
            return None

    def __del__(self):
        pygame.quit()
