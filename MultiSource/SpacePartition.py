
import sys
sys.path.append('..')
import argparse
from MultiSource.MultiBase import *

class SpacePartition(MultiBase):
    def __init__(self, sources = None, mode = 1, expname = ''):
        '''mode的值不再产生影响'''
        MultiBase.__init__(self, sources = sources, expname = expname)

    def Exploration(self):
        stime = datetime.datetime.now()
        ss = None
        while ss is None:
            for src in self.srcs:
                bMerge = self.walk(src)
                if src.isstart and src.isend:
                    ss = src # src包含起始点
                    break
                elif bMerge:
                    time.sleep(10)
                    break
        self.wallclock_expand = (datetime.datetime.now() - stime).total_seconds()
        self.episodes_expand = sum([x.expand_episodes for x in self.srcs])
        self.finalsource = ss
        self.inner_train()
        if len(self.srcs) > 1:
            print('没有完全合并，需要处理')

    def walk(self, src : Source):
        action = src.agent.choose_action(encode(src.cur))
        reward, done, info, next = super().step(src, action)

        bMerge = False
        if info == WALK:# 尝试合并
            bMerge = self.trymerge(src, next)

        src.agent.learn(encode(src.cur), action, reward, encode(next), done)

        src.steps = (1 + src.steps) % EXPLORATION_HORIZON
        src.cur = next
        self.move_block(src, next)

        # 碰撞（出界）或抵达终点，或满EXPLORATION_HORIZON步，episode结束
        if done or src.steps == 0:
            src.cur = src.start
            src.expand_episodes += 1
            self.move_block(src, src.cur)

        return bMerge

    def trymerge(self, src : Source, next):
        # 如果自己走过就不合并了
        if src.contain(next):
            return False
        target = self.intersection(next, src.name)
        # 别人也没走过
        if target is None:
            src.append(next)
            self.add_block(src, next)
            return False
        # 如果走进别人的地盘，先确定谁吃谁
        if src.end is None and target.end is None:
            #.ff_merge()
            flag = closer(src.start, target.start, self.destination)
            eater, food = (target, src) if flag else (src, target)
            eater.end = food.start
        elif src.end is not None and target.end is not None:
            #.tt_merge()
            flag = closer(src.start, target.start, self.start)
            eater, food = (src, target) if flag else (target, src)
            flag = closer(src.end, target.end, self.destination)
            eater.end = src.end if flag else target.end
        else:
            #.tf_merge()
            flag = closer(src.start, target.start, self.destination)
            eater, food = (target, src) if flag else (src, target)
            if eater.end is None:
                eater.end = food.end
            elif closer(food.start, eater.end, self.destination):
                eater.end = food.start

        #print('{} {} merges {} at {}'.format(datetime.datetime.now(), eater.name, food.name, src.cur))
        eater.cur = src.cur
        eater.points.extend(food.points)
        eater.name = '{}to{}'.format(eater.start, eater.end)
        eater.isstart |= food.isstart
        eater.isend |= food.isend
        #eater.steps = 0
        # 合并qtable
        eater.agent.q_table = self.merge_qtable(eater.agent.q_table, food.agent.q_table)

        src = eater
        src.expand_episodes = eater.expand_episodes + food.expand_episodes
        src.steps = eater.steps + food.steps
        if src.steps >= EXPLORATION_HORIZON:
            src.expand_episodes += 1
        self.srcs.remove(food)
        #self.display()
        #self.inner_train(eater)
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
            guide_table(src.agent.q_table, self.height, self.width, '{} {} {}'.format(self.expname, length, src.name))

    def inner_train(self):
        self.finalsource.steps = 0
        self.finalsource.cur = self.start
        self.finalsource.isend = False
        self.finalsource.expand_episodes = 0
        stime = datetime.datetime.now()
        while not self.finalsource.isend:
            bMerge = self.walk(self.finalsource)
        self.wallclock_inner = (datetime.datetime.now() - stime).total_seconds()
        self.episodes_inner = self.finalsource.expand_episodes

'''有结果马上写文档里'''
def test(srcs):
    fname = SPaRMname(1, srcs)
    for _ in range(100):
        ms = SpacePartition(srcs)
        ms.merge()
        td2 = ms.inner_train()
        with open('../img/{}merge.txt'.format(fname), 'a', encoding='utf-8') as f:
            f.write('{},\n'.format([ms.mergetime.total_seconds(), td2.total_seconds()]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1)
    args = parser.parse_args()
    n = args.n

    SpacePartition(srcdata[0]).Exploration()

    #test(srcdata[n])
    #STEP_REWARD = -0.01
    #test1(srcdata[n], mode )
