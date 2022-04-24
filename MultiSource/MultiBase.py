import sys
sys.path.append('..')
import datetime
import matplotlib.pyplot as plt

from Common.dmdp_actions import *
from Maze.ComplexMaze import *
from Source import *

OUT = 'out of bound'
CRASH = 'collision'
ARRIVE = 'arrive'
WALK = 'walk'
MERGE = 'merge'
FOUND = 'found'

ARRIVE_REWARD = 0#1#
CRASH_REWARD = -1.0
STEP_REWARD = 0#-0.01#

END_IF_OUT = True # 出界时是否结束训练

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
            Source('start', self.start, isstart = True),
            #Source('10-7', np.array([10, 7])),
            Source('8-2', np.array([8, 2])),
            Source('13-10', np.array([13, 10])),
            #Source('15-14', np.array([15, 14])),
            Source('end', self.destination, isend = True)
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
            info = self.iterstep()
            if info == FOUND:
                etime = datetime.datetime.now()
                print(etime, 'found. total', etime - stime)
                break
        #gt = guide_table(self.agent.q_table, self.height, self.width, 'global')
        #plt.show()
        #print(gt)
    def iterstep(self):
        '''迭代，让每个source走一步'''
        for src in self.srcs:#[0: -1]:#
            info = self.walk(src)
            if info == MERGE or info == FOUND:
                self.display()
                return info

    def walk(self, pos, action): # walk(self, src)
        '''每个source走一步'''
        pass
    def merge(self, walker, other):
        '''合并两个source'''
        pass
    def display(self):
        pass
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
