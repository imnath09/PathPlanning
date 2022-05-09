
import numpy as np
import matplotlib.pyplot as plt

from Common.dmdp_enum import *

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
