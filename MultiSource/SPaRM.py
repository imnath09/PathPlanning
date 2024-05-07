
from datetime import timedelta
import sys
sys.path.append('..')
import argparse
from MultiSource.MultiBase import *

class SPaRM(MultiBase):
    def __init__(self, sources = None, mode = 1, expname = ''):
        '''mode的值不再产生影响'''
        MultiBase.__init__(self, sources = sources, expname = expname)
        self.finalsource = Source(self.destination, isend = True)

    def Exploration(self):
        for src in self.srcs:
            srctime = datetime.datetime.now()
            while not self.walk(src):
                continue
            src.expand_episodes = src.inner_episodes
            src.inner_episodes = self.finalsource.inner_episodes
            src.expand_time = datetime.datetime.now() - srctime -src.inner_time
            time.sleep(10)
        self.wallclock_expand = sum([x.expand_time.total_seconds() for x in self.srcs])
        self.wallclock_inner = sum([x.inner_time.total_seconds() for x in self.srcs])
        self.episodes_expand = sum([x.expand_episodes for x in self.srcs])
        self.episodes_inner = sum([x.inner_episodes for x in self.srcs])
        #self.display_all()

    def walk(self, src : Source, inner = False):
        action = src.agent.choose_action(encode(src.cur))
        reward, done, info, next = super().step(src, action, inner)
        src.agent.learn(encode(src.cur), action, reward, encode(next), done)

        src.steps = (1 + src.steps) % EXPLORATION_HORIZON
        src.cur = next
        self.move_block(src, next)

        # 是否发生合并，或者抵达终点
        if info == WALK:# 尝试和finalsource合并
            if self.trymerge(src, next):
                src.inner_episodes += 1
                return True
        elif info == ARRIVE or info == FOUND:
            self.trymerge(src, next)
            src.inner_episodes += 1
            return True

        # 碰撞（出界）或者满EXPLORATION_HORIZON步，结束episode
        if done or src.steps == 0:
            src.cur = src.start
            src.inner_episodes += 1
            self.move_block(src, src.cur)

        return False

    def trymerge(self, src : Source, next):
        '''尝试和finalsource源合并'''
        # 如果自己走过就不合并了
        if src.contain(next):
            return False

        # 如果还没有终点源，或者没进入终点源，不用合并
        if (not self.finalsource.contain(next)):#(self.finalsource is None) or 
            src.append(next)
            self.add_block(src, next)
            return False

        # 如果走进终点源的地盘，被终点源吃掉
        self.finalsource.cur = src.start
        self.finalsource.points += src.points
        self.finalsource.start = src.start
        self.finalsource.steps = 0
        self.finalsource.inner_episodes = 0
        # 合并qtable
        self.finalsource.agent.q_table = self.merge_qtable(self.finalsource.agent.q_table, src.agent.q_table)

        #self.display(src)
        innertime = self.inner_explore(self.finalsource)
        src.inner_time = innertime
        return True

    def inner_explore(self, src : Source):
        stime = datetime.datetime.now()
        while not self.walk(src, True):
            continue
        return datetime.datetime.now() - stime

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

    def display_all(self):
        for src in self.srcs + [self.finalsource]:
            guide_table(src.agent.q_table, self.height, self.width, '{} {}'.format(self.expname, src.name))

    def display(self, src):
        guide_table(src.agent.q_table, self.height, self.width, src.name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, )
    parser.add_argument('--mode', type=int, default=0, help='0随机 1不随机')
    args = parser.parse_args()
    mode = args.mode
    n = args.n

    rm = SPaRM(sources=srcdata[1],)
    rm.Exploration()

    #test1(srcdata[n], mode)
    #STEP_REWARD = -0.01
    #test1(srcdata[n], mode )
