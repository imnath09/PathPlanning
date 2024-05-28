
import sys
sys.path.append('..')

from Algs.SpacePartition import *


class SPaRM(SpacePartition):
    def __init__(self, agent, env, n_walkers=3):
        SpacePartition.__init__(self, agent, env, n_walkers)