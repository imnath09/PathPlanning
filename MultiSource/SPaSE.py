
from datetime import timedelta
import sys
sys.path.append('..')
import argparse
from MultiSource.MultiBase import *

class SPaSE(MultiBase):
    def __init__(self, sources = None, mode = 0, expname = ''):
        '''params:
        mode: 0-随机跳转; 1-不随机跳转;
        '''
        MultiBase.__init__(self, sources = sources, mode = mode, expname = expname)
        #self.exploretime = timedelta(seconds = 0)

    def merge(self):
        stime = datetime.datetime.now()
        for src in self.srcs:
            while not self.walk(src):
                continue
        mergetime = datetime.datetime.now() - stime
        #print(mergetime, 'finish')
        #self.display()
        return mergetime

    def walk(self, src : Source, inner = False):
        action = src.agent.choose_action(encode(src.cur))
        reward, done, info, next = super().step(src, action, inner)
        src.agent.learn(encode(src.cur), action, reward, encode(next), done)

        src.steps = (1 + src.steps) % self.jumpgap
        src.cur = next
        self.move_block(src, next)

        mrg = False
        if info == WALK:# 尝试合并
            mrg = self.trymerge(src, next)
        elif info == ARRIVE:
            self.finalsource = src
            mrg = True

        # 碰撞（出界）或者满jumpgap步，随机跳
        if done or src.steps == 0: # 碰撞（出界），但没抵达终点。这些情况下随机跳
            if self.mode == 0:
                src.rjump()
            else:
                src.cur = src.start
            self.move_block(src, src.cur)

        return mrg

    def trymerge(self, src : Source, next):
        # 如果自己走过就不合并了
        if src.contain(next):
            return False

        # 如果还没有终点源，或者没进入终点源，不用合并
        if (self.finalsource is None) or (not self.finalsource.contain(next)):
            src.append(next)
            self.add_block(src, next)
            return False

        # 如果走进终点源的地盘，被终点源吃掉
        self.finalsource.cur = src.cur
        self.finalsource.points += src.points
        self.finalsource.start = src.start
        # 合并qtable
        self.finalsource.agent.q_table = self.merge_qtable(self.finalsource.agent.q_table, src.agent.q_table)

        #self.display()
        self.inner_explore(self.finalsource)
        return True

    def merge_qtable(self, eater, droper):
        its = list(set(eater.index).intersection(set(droper.index)))
        #print(its)
        if len(its) > 0:
            #print('intersection', its)
            droper.drop(index = its, inplace = True)
        its = list(set(eater.index).intersection(set(droper.index)))
        if len(its) > 0:
            print('after', its)
        return pd.concat([eater, droper])

    def display(self):
        length = str(len(self.srcs))
        for src in self.srcs:
            guide_table(src.agent.q_table, self.height, self.width, '{} {}'.format(self.expname, src.name))

    def inner_explore(self, src : Source):
        return datetime.timedelta(seconds=0)
        stime = datetime.datetime.now()
        while True:
            action = src.agent.choose_action(encode(src.cur))
            reward, done, info, next = super().step(src, action, True)
            src.agent.learn(encode(src.cur), action, reward, encode(next), done)

            self.move_block(src, next)

            if info == FOUND or info == ARRIVE:
                etime = datetime.datetime.now()
                self.exploretime += etime - stime
                return
            src.steps = (1 + src.steps) % self.jumpgap
            if done or src.steps == 0:
                src.cur = src.start
                self.move_block(src, src.cur)
            else:
                src.cur = next


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, )
    parser.add_argument('--mode', type=int, default=0, help='0随机 1不随机')
    args = parser.parse_args()
    mode = args.mode
    n = args.n

    #test1(srcdata[n], mode)
    #STEP_REWARD = -0.01
    #test1(srcdata[n], mode )
