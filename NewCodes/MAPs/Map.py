import sys
sys.path.append('..')

from .InitMap import *
from .ComplexMaze import *
from Common.dmdp_enum import *

PIXEL = 30
OUT = 'out of bound'
CRASH = 'collision'
ARRIVE = 'arrive'
WALK = 'walk'

END_IF_OUT = True # 出界时是否结束训练
END_IF_CRASH = True # 碰撞时是否结束训练

map1 = np.array([
    [0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,],
    [0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,],
    [1,  1,  1,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  1,  1,  0,  0,],
    [0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,],
    [0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,],
    [0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,],
    [1,  1,  1,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,],
    [0,  0,  0,  0,  0,  1,  1,  1,  1,  0,  0,  1,  0,  0,  0,  0,  0,],
    [0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  1,  1,  0,  0,],
    [0,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,],
    [0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,],
    [0,  0,  0,  0,  0,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  0,  0,],
    [0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,],
    [1,  1,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,],
    [0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  1,  1,  1,  1,  0,  0,],
    [0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,],
    [0,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,],
    [0,  0,  0,  0,  0,  1,  0,  1,  1,  1,  0,  1,  0,  0,  1,  1,  1,],
    [0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,],
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,]])

#from numpy import array
#os = [array([0, 5]), array([ 0, 14]), array([1, 5]), array([ 1, 14]), array([2, 0]), array([2, 1]), array([2, 2]), array([2, 5]), array([ 2, 11]), array([ 2, 12]), array([ 2, 13]), array([ 2, 14]), array([3, 5]), array([3, 8]), array([ 3, 11]), array([4, 5]), array([4, 8]), array([ 4, 11]), array([5, 8]), array([5, 9]), array([ 5, 10]), array([ 5, 11]), array([ 5, 14]), array([ 5, 15]), array([ 5, 16]), array([6, 0]), array([6, 1]), array([6, 2]), array([6, 8]), array([ 6, 11]), array([7, 5]), array([7, 6]), array([7, 7]), array([7, 8]), array([ 7, 11]), array([8, 5]), array([ 8, 11]), array([ 8, 12]), array([ 8, 13]), array([ 8, 14]), array([9, 2]), array([9, 3]), array([9, 4]), array([9, 5]), array([ 9, 14]), array([10,  5]), array([10, 14]), array([11,  5]), array([11,  8]), array([11,  9]), array([11, 10]), array([11, 11]), array([11, 12]), array([11, 13]), array([11, 14]), array([12,  8]), array([13,  0]), array([13,  1]), array([13,  2]), array([13,  8]), array([14,  5]), array([14,  8]), array([14, 11]), array([14, 12]), array([14, 13]), array([14, 14]), array([15,  5]), array([15, 11]), array([16,  2]), array([16,  3]), array([16,  4]), array([16,  5]), array([16, 11]), array([17,  5]), array([17,  7]), array([17,  8]), array([17,  9]), array([17, 11]), array([17, 14]), array([17, 15]), array([17, 16]), array([18,  5]), array([18, 11]), array([19, 11])]

class MapBase():
    def __init__(self, map = map1):
        self.s_path = []
        self.cur_path = []
        self.new_sln = False
        self.action_space = range(4)  #list(actions)[:4]
        #self.action_space_n = len(self.action_space)
        #self.observation_space_n = 2

        self.parse_map(map)

        self.ARRIVE_REWARD = 2.0
        self.CRASH_REWARD = -1.0
        self.STEP_REWARD = -0.001
        self.MERGE_REWARD = 0

        #print(REWARD)

    def parse_map(self, map):
        self.map = map
        self.obstacles = []
        self.height, self.width = map.shape
        self.start = np.array([0, 0])
        self.walker = self.start
        self.destination = np.array([self.height - 1, self.width - 1])
        for i in range(self.height):
            for j in range(self.width):
                if map[i][j] == 1:
                    self.obstacles.append(np.array([i, j]))


    def reset(self):
        self.walker = self.start
        self.cur_path = []
        return self.walker

    def step(self, action : int):
        self.new_sln = False
        next = self.walker + DIRECTION[action]
        self.cur_path.append(next)
        info = None
        # 出界
        if (next[0] < 0 or
            next[1] < 0 or
            next[0] >= self.height or
            next[1] >= self.width):
            reward = self.CRASH_REWARD
            done = END_IF_OUT # 出界是否结束
            #print(OUT)
            if not END_IF_OUT:
                next = self.walker # 出界不结束就回退
            info = OUT
        # 碰撞
        elif any((next == x).all() for x in self.obstacles):
            reward = self.CRASH_REWARD
            done = END_IF_CRASH # 碰撞是否结束
            if not END_IF_CRASH:
                next = self.walker
            info = CRASH
        # 抵达终点
        elif (next == self.destination).all():
            reward = self.ARRIVE_REWARD
            #print('daoda{}'.format(reward))
            done = True
            info = ARRIVE

            if (len(self.cur_path) < len(self.s_path) or
                self.s_path.__len__() == 0):
                self.new_sln = True
            #elif len(self.cur_path) == len(self.s_path):
            #    self.new_sln = True
            #    for i in range(len(self.s_path)):
            #        if not str(self.cur_path[i]) == str(self.s_path[i]):
            #            self.new_sln = False
            #            break
            else: # >
                self.new_sln = False

            if self.new_sln:
                self.s_path = self.cur_path.copy()
                #info = "path length:{}".format(self.cur_path.__len__())
                #for c in self.cur_path:
                #    info = ("{}->{}".format(info, str(c)))
        # 移动
        else:
            '''
            a:walker->next
            b:walker->destination
            c:next->destination
            cosine->a**2 + b**2 - c**2 / 2ab
            a2 = (self.walker[0] - next[0])**2 + (self.walker[1] - next[1])**2
            b2 = (self.walker[0] - self.map.destination[0])**2 + (self.walker[1] - self.map.destination[1])**2
            c2 = (self.map.destination[0] - next[0])**2 + (self.map.destination[1] - next[1])**2
            if a2 == 0 or b2 == 0:
                cosine = 0
            else:
                cosine = (a2 + b2 - c2) / (2 * (a2 * b2)**0.5)
            reward = cosine - math.log(len(self.cur_path), 100)
            '''
            reward = self.STEP_REWARD#REWARD[tuple(next)]#
            #print('cos=', cosine)
            info = WALK
            done = False

        self.walker = next
        return next, reward, done, info

    def render(self):
        pass








