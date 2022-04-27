import sys
sys.path.append('..')
import datetime
import matplotlib.pyplot as plt

from Common.dmdp_enum import *
from Maze.ComplexMaze import *
from MultiSource.Source import *

OUT = 'out of bound'
CRASH = 'collision'
ARRIVE = 'arrive'
WALK = 'walk'
MERGE = 'merge'
FOUND = 'found'

ARRIVE_REWARD = 1#0#
CRASH_REWARD = -1.0
STEP_REWARD = 0#-0.01#

END_IF_OUT = False # 出界时是否结束训练

class MultiBase:
    def __init__(self):
        self.map = CplxMaze()
        self.height = self.map.height
        self.width = self.map.width
        self.obstacles = self.map.obstacles
        self.destination = self.map.destination
        self.start = self.map.start
        self.agent = None
        self.srcs = [
            Source(self.start, isstart = True),
            Source(np.array([8, 2])),
            Source(np.array([10, 7])),
            #Source(np.array([13, 10])),
            Source(np.array([15, 14]))
            #Source(self.destination, isend = True)
            ]
    def intersection(self, src : Source):
        for s in self.srcs:
            if not s.name == src.name and s.contain(src.cur):
                return s
        return None
    def intersection(self, pos, name):
        for s in self.srcs:
            if not s.name == name and s.contain(pos):
                return s
        return None
    def explore(self):
        print(len(self.srcs), 'sources')
        stime = datetime.datetime.now()
        print(stime, 'start')
        while True:
            result = self.iterstep()
            if result is not None:
                etime = datetime.datetime.now()
                print(etime, 'found. total', etime - stime)
                return result
    def iterstep(self):
        '''迭代，让每个source走一步'''
        for src in self.srcs:#[0: -1]:#
            result = self.walk(src)
            if result.isstart and result.isend:
                return result
        return None
    def walk(self, src):
        '''每个source走一步'''
        pass

    def step(self, src, action):
        next = src.cur + DIRECTION[action]
        # 出界
        if (next[0] < 0 or
            next[1] < 0 or
            next[0] >= self.height or
            next[1] >= self.width):
            reward = CRASH_REWARD
            done = END_IF_OUT # 出界是否结束
            if not END_IF_OUT:
                next = src.cur # 出界结束就随机跳，否则回退
            info = OUT
        # 碰撞
        elif any((next == x).all() for x in self.obstacles):
            reward = CRASH_REWARD
            done = True
            info = CRASH
        # 抵达目的地
        elif (next == self.destination).all():
            src.end = next
            src.isend = True
            reward = ARRIVE_REWARD
            done = True
            info = FOUND if src.isstart else ARRIVE
        # 抵达源内部终点
        elif (next == src.end).all():
            reward = ARRIVE_REWARD / 1000
            done = True
            info = ARRIVE
        # 正常移动
        else:# 
            done = False
            info = WALK
            reward = STEP_REWARD

        return reward, done, info, next

    def add_block(self, src, pos):
        pass
    def move_block(self, src, pos):
        pass

def encode(pos):
    '''编成字符'''
    r = '{},{}'.format(pos[0], pos[1])
    return r
def decode(index):
    l = index.split(',')
    for i in range(len(l)):
        l[i] = int(l[i])
    l = np.array(l)
    return l
def ops(action):
    if action == 0:
        return 1
    elif action == 1:
        return 0
    elif action == 2:
        return 3
    elif action == 3:
        return 2
    else:
        return 4
def guide_table(table, height, width, title):
    '''画策略图'''
    ntbl = np.full((height + 2, width + 2), 4.0)
    for r in table.index:
        pos = tuple(decode(r) + [1, 1])
        s = table.loc[r]
        c = np.random.choice(s[s==np.max(s)].index)
        content = actions(c).name[0]#.ljust(5, ' ')
        ntbl[pos] = c
        plt.annotate(text=content, xy=(pos[1], pos[0]), ha='center', va='center')
    plt.title(title)
    plt.imshow(ntbl, cmap='Greens_r', vmin = 0, vmax = 4)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('../img/{}.png'.format(title))
    plt.close()
    return ntbl
def euclidean2(pos1, pos2):
    return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
def closer(p1, p2, dest):
    '''p1是否比p2更接近dest'''
    if p1 is None and p2 is None:
        return None
    if p1 is None:
        return False
    if p2 is None:
        return True
    d1 = euclidean2(p1, dest)
    d2 = euclidean2(p2, dest)
    return d1 < d2
