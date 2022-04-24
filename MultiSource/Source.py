
from Algorithm.QLearning import *

class Source():
    def __init__(self, name, pos, isstart = False, isend = False):
        self.cur = pos
        self.points = [self.cur]
        self.name = name
        self.isstart = isstart
        self.isend = isend
        self.steps = 0
        self.agent = QLearningTable(actions = range(4), e_greedy = 0.9)
        self.ragent = QLearningTable(actions = range(4), e_greedy= 0.9)
        #self.forward = True
        self.inner_start = None
        self.inner_end = None

    def tryappend(self, pos):
        if not self.contain(pos):
            self.points.append(pos)
            return True
        return False
    def append(self, pos):
        self.points.append(pos)
    def contain(self, pos):
        for p in self.points:
            if (pos == p).all():
                return True
        return False
    def rjump(self):
        rp = np.random.randint(0, len(self.points))
        self.cur = self.points[rp]
