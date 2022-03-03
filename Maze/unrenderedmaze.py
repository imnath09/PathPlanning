import numpy as np
import sys

'''
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
'''
sys.path.append('..')
from Common.dmdp_actions import *

PIXEL = 50
MAZE_HEIGHT = 8
MAZE_WIDTH = 8
OUT = 'out of bound'
CRASH = 'collision'

ARRIVE_REWARD = 10.0
CRASH_REWARD = -1.0
STEP_REWARD = -0.01

END_IF_OUT = False # 出界时是否结束训练

START = np.array([0, 0]) # 起始点
DESTINATION = np.array([2, 2]) # 终点

'''
REWARD = np.array(
[[ 4.94,  5.26,  4.2 ,  5.12,  2.74,  2.07,  1.49,  1.26],
 [ 4.96,  6.04,  1.72,  1.7 ,  1.23,  2.95,  3.29,  1.86],
 [ 2.43,  0.  ,  10  ,  1.67,  2.79,  3.03,  0.89,  0.77],
 [ 1.52,  2.23,  7.79,  6.59,  2.23,  1.45,  1.34,  0.71],
 [ 3.26,  4.06,  1.29,  1.48,  1.1 ,  2.2 ,  2.49,  0.8 ],
 [ 3.56,  3.63,  3.95,  2.45,  2.27,  1.63,  1.14,  0.48],
 [ 3.63,  4.  ,  1.67,  3.32,  3.66,  1.57,  1.07,  0.37],
 [ 1.37,  1.66,  3.97,  3.83,  1.97,  1.58,  1.03,  0.15]])
REWARD = (REWARD - 10.0) / 1000
'''

class UnrenderedMaze():
    def __init__(self):
        self.s_path = []
        self.cur_path = []
        self.action_space = range(4)  #list(actions)[:4]
        self.action_space_n = len(self.action_space)
        self.observation_space_n = 2
        self.new_sln = False

        self.walker = START
        self.obstacles = [
            np.array([2, 1]),
            np.array([1, 2]),
            np.array([2, 6]),
            np.array([6, 2]),
            np.array([4, 4]),
            np.array([4, 1]),#4 1
            #np.array([1, 3]),
            np.array([2, 4]),#2 4
            #np.array([3, 2]),#3 2
        ]

        #print(REWARD)

    def reset(self):
        self.walker = START
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
            next[0] >= MAZE_HEIGHT or
            next[1] >= MAZE_WIDTH):
            reward = CRASH_REWARD
            done = END_IF_OUT # 出界是否结束
            if not END_IF_OUT:
                next = self.walker # 出界不结束就回退
            info = OUT
        # 碰撞
        elif any((next == x).all() for x in self.obstacles):
            reward = CRASH_REWARD
            done = True
            info = CRASH
        # 抵达终点
        elif (next == DESTINATION).all():
            reward = ARRIVE_REWARD
            #print('daoda{}'.format(reward))
            done = True

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
            reward = STEP_REWARD#REWARD[tuple(next)]#
            done = False

        self.walker = next
        return next, reward, done, info

    def render(self):
        pass








