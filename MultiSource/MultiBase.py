import sys
sys.path.append('..')
import datetime

from Maze.ComplexMaze import *
from MultiSource.Source import *
from Common.utils import *

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
END_IF_CRASH = False # 碰撞时是否结束训练

class MultiBase:
    def __init__(self):
        self.map = CplxMaze()
        self.height = self.map.height
        self.width = self.map.width
        self.obstacles = self.map.obstacles
        self.destination = self.map.destination
        self.start = self.map.start
        #self.agent = None
        self.srcs = [
            Source(self.start, isstart = True),
            Source(np.array([8, 2])),
            Source(np.array([10, 7])),
            #Source(np.array([13, 10])),
            Source(np.array([15, 14]))
            #Source(self.destination, isend = True)
            ]
        # 预留着以后合并UnrenderdMaze进来
        self.s_path = []
        self.cur_path = []
        self.action_space = range(4)  #list(actions)[:4]
        self.action_space_n = len(self.action_space)
        self.observation_space_n = 2
        self.new_sln = False
    # 预留着以后合并UM
    def reset(self):
        pass
    def render(self):
        pass
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
    def walk(self, src : Source):
        '''每个source走一步'''
        pass

    def step(self, src : Source, action):
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
            done = END_IF_CRASH # 碰撞是否结束
            if not END_IF_CRASH:
                next = src.cur
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
