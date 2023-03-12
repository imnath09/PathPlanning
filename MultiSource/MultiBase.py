import sys
sys.path.append('..')
import datetime

from Maze.unrenderedmaze import *
from Maze.ComplexMaze import *
from MultiSource.Source import *
from Common.utils import *

WALK = 'walk'
MERGE = 'merge'
FOUND = 'found'

EXPLORATION_HORIZON = 500
PLANNING_HORIZON = 60

p1 = np.array([15, 14])
p2 = np.array([15, 8])
p3 = np.array([10, 7])
p4 = np.array([12, 5])
p5 = np.array([8, 2])#比较废的点

srcdata = [
    [p1, p2, p3], # 0
    [p1, p2, p4], # 1
    [p1, p2, p5], # 2
    [p1, p3, p4], # 3
    [p1, p3, p5], # 4
    [p1, p4, p5], # 5
    [p1, p5],
    [p1, p2],
    [p1, p4],
    [p1, p3],
    [p1], # 8
    [p2],
    [p3],
    [p4],
    [], # 12
]

def SPaRMname(mode, srcs):
    if mode == 2:
        return 'QLearning'
    info = '_'.join(['{}{}'.format(x[0], x[1]) for x in srcs])
    if mode == 1:
        return 'SP{}_{}'.format(len(srcs) + 1, info)
    elif mode == 0:
        return 'SPaRM{}_{}'.format(len(srcs) + 1, info)
    elif mode == 3:
        return 'RFE'

def SPaSEname(mode, srcs):
    if mode == 0 and len(srcs) == 0:
        return 'SE'
    elif mode == 0 and len(srcs) > 0:
        n = 'SPaSE{}'.format(len(srcs) + 1)
        for s in srcs:
            n += '_{}{}'.format(s[0], s[1])
        return n
    elif mode == 1 and len(srcs) == 0:
        return 'RFE'
    elif mode == 1 and len(srcs) > 0:
        n = 'SP{}'.format(len(srcs) + 1)
        for s in srcs:
            n += '_{}{}'.format(s[0], s[1])
        return n
    else:
        return 'QLearning'

class MultiBase(UnrenderedMaze):
    def __init__(self, sources = None, mode = 0, expname = ''):
        UnrenderedMaze.__init__(self)
        self.destination = self.map.destination
        self.start = self.map.start
        self.mode = mode
        self.expname = expname
        self.finalsource = None
        self.exploretime = datetime.timedelta(seconds = 0)
        self.mergetime = datetime.timedelta(seconds = 0)
        if sources == None:# or len(sources) == 0:
            self.srcs = []
        else:
            self.srcs = [Source(x) for x in sources]
        self.srcs.append(Source(self.start, isstart = True))
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
    def intersection_around(self, pos, name):
        for s in self.srcs:
            if not s.name == name and s.contain_any([x + pos for x in DIRECTION]):
                return s
        return None

    def merge(self):
        #print(len(self.srcs), 'sources')
        stime = datetime.datetime.now()
        #print(stime, 'start')
        while True:
            source = self.iterstep()
            if source is not None:
                #self.srcs = [source]
                self.finalsource = source
                etime = datetime.datetime.now()
                time_delta = etime - stime
                #print(etime, 'found. total', time_delta)
                self.mergetime = time_delta#result
                return time_delta#result
    def iterstep(self):
        '''迭代，让每个source走一步'''
        for src in self.srcs:#[0: -1]:#
            source = self.walk(src)
            if source.isstart and source.isend:
                return source
        return None
    def walk(self, src : Source, inner = False):
        '''每个source走一步'''
        pass

    def step(self, src : Source, action, inner = False):
        next = src.cur + DIRECTION[action]
        # 出界
        if (next[0] < 0 or
            next[1] < 0 or
            next[0] >= self.height or
            next[1] >= self.width
            ) or (
                inner and not src.contain(next)
            ):
            reward = self.CRASH_REWARD
            done = END_IF_OUT # 出界是否结束
            if not END_IF_OUT:
                next = src.cur # 出界结束就随机跳，否则回退
            info = OUT
        # 碰撞
        elif any((next == x).all() for x in self.obstacles):
            reward = self.CRASH_REWARD
            done = END_IF_CRASH # 碰撞是否结束
            if not END_IF_CRASH:
                next = src.cur
            info = CRASH
        # 抵达目的地
        elif (next == self.destination).all():
            src.end = next
            src.isend = True
            reward = self.ARRIVE_REWARD
            done = True
            info = FOUND if src.isstart else ARRIVE
        # 抵达源内部终点
        elif (next == src.end).all():
            reward = self.MERGE_REWARD
            done = False
            info = ARRIVE
        # 正常移动
        else:# 
            done = False
            info = WALK
            reward = self.STEP_REWARD

        return reward, done, info, next

    def add_block(self, src, pos):
        pass
    def move_block(self, src, pos):
        pass
