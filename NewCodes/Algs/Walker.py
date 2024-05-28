
import random

class Walker():
    def __init__(self, start, isstart = False):
        self.cur = start
        self.points = [start]
        self.steps = 0
        self.start = start
        self.name = str(start)

        self.isstart = isstart
        self.isend = False
        self.steps = 0


    def contain(self, position):
        for p in self.points:
            if (position == p).all():
                return True
        return False

    def append(self, position):
        self.points.append(position)

    def jump(self):
        self.cur = random.choice(self.points)
        return self.cur




