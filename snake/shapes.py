class Shape:

    def __init__(self, block_size, shape='rect'):
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
            window.brush(self.shape, self.color, rect, 0)

    def _c2p(self, pos, window):
        return window.base[0] + pos[0]*self.block_size, window.base[1] + pos[1]*self.block_size

    @staticmethod
    def _tuple_add(t1, t2):
        return t1[0] + t2[0], t1[1] + t2[1]


class Rect(Shape):
    pass


class Circle(Shape):

    def __init__(self, **kwargs):
        super().__init__(shape='circle', **kwargs)

    def draw(self, window):
        if not isinstance(self.pos, list):
            pos = [self.pos]
        else:
            pos = self.pos
        for p in pos:
            p = self._c2p(p, window)
            window.brush(self.shape, self.color, p, self.block_size // 2)

    def _c2p(self, pos, window):
        base = super()._c2p(pos, window)
        return base[0] + self.block_size // 2, base[1] + self.block_size // 2


class Frame(Rect):

    def __init__(self, **kwargs):
        self.color = kwargs.pop('color')
        super().__init__(**kwargs)
        self.pos = (0, 0)
