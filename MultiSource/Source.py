
from Algorithm.QLearning import *

class Source():
    def __init__(self, start, end = None, name = None, isstart = False, isend = False):
        self.cur = start
        self.points = [start]
        self.start = start
        self.end = end
        self.name = name if name is not None else str(start)
        self.isstart = isstart
        self.isend = isend
        self.steps = 0
        self.agent = QLearningTable(actions = range(4), e_greedy = 0.9)
        #self.ragent = QLearningTable(actions = range(4), e_greedy= 0.9)

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
    def contain_any(self, poss):
        for p in self.points:
            if any((x == p).all() for x in poss):
                return True
        return False
    def rjump(self):
        rp = np.random.randint(0, len(self.points))
        self.cur = self.points[rp]
