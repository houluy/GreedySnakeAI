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

        self.wbcolor = (84, 255, 159)
        self.wfcolor = (46, 139, 87)

        self.bcolor = (151, 255, 255)
        self.fcolor = (82, 139, 139)

        self.hbcolor = (0, 191, 255)
        self.hfcolor = (0, 0, 205)

        self.bbcolor = (132, 112, 255)  # body bcolor
        self.bfcolor = (72, 61, 139)  # body fcolor

        self.fbcolor = (255, 250, 205)  # food bcolor
        self.ffcolor = (255, 222, 173)  # food fcolor

        self.inner = 0.1
        self.iblock = int(self.block_size * (1 - 2 * self.inner))

        self.screen = pygame.display.set_mode(self.background_shape)
        pygame.display.set_caption('Greedy Snake AI')
        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill(self.background_color)

        self.colors = {
            0: (self.fcolor, self.bcolor, "rect"),
            1: (self.hfcolor, self.hbcolor, "rect"),  # head
            -1: (self.bfcolor, self.bbcolor, "rect"),  # body
            2: (self.ffcolor, self.fbcolor, "circle"),  # food
            -2: (self.wfcolor, self.wbcolor, "rect"),  # wall
        }

        self.directions = dict(zip([pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN], range(4)))

        self.brushes = {
            'rect': pygame.draw.rect,
            'circle': pygame.draw.circle,
        }

    def _c2p(self, pos, shift=0):
        shift = self.block_size * shift
        return self.base[0] + pos[0]*self.block_size + shift, self.base[1] + pos[1]*self.block_size + shift

    def _c2c(self, pos):
        return self.base[0] + pos[0] * self.block_size + self.block_size // 2,\
               self.base[1] + pos[1] * self.block_size + self.block_size // 2

    def draw(self, state):
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()
        for ind, val in np.ndenumerate(state):
            fc, bc, shape = self.colors[val]
            if shape == 'rect':
                co = (*self._c2p(ind), self.block_size, self.block_size)
                sco = (*self._c2p(ind, shift=self.inner), self.iblock, self.iblock)
                pygame.draw.rect(self.background, bc, co)
                pygame.draw.rect(self.background, fc, sco)
            elif shape == 'circle':
                center, radius = self._c2c(ind), self.block_size // 2
                scenter, sradius = self._c2c(ind), int(self.iblock // 2)
                pygame.draw.circle(self.background, bc, center, radius)
                pygame.draw.circle(self.background, fc, scenter, sradius)

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

    # def brush(self, shape, color, co, width=0):
    #     if shape == 'rect':
    #         pygame.draw.rect(self.background, color, co, width)
    #     elif shape == 'circle':
    #         pygame.draw.circle(self.background, color, )

    def __del__(self):
        pygame.quit()
