


import sys
sys.path.append('..')

from MultiSource.SpacePartition import *


class RFE(SpacePartition):
    def __init__(self, sources = None, mode = 1, expname = ''):
        SpacePartition.__init__(self,sources=sources,expname=expname)
        self.STEP_REWARD = 0




