
import sys
sys.path.append('..')
import argparse
from MultiSource.MultiBase import *

class MultipleReversal(MultiBase):
    def __init__(self, sources = None, mode = 0, expname = ''):
        '''params:
        mode: 0-随机跳转; 1-不随机跳转;
        '''
        MultiBase.__init__(self, sources = sources)
        self.mode = mode
        self.expname = expname
        self.jumpgap = 100 if mode == 0 else 500

    def walk(self, src : Source):
        action = src.agent.choose_action(encode(src.cur))
        reward, done, info, next = super().step(src, action)

        if info == WALK:# 尝试合并
            src = self.trymerge(src, next)

        src.agent.learn(encode(src.cur), action, reward, encode(next), done)
        #if reward > 0:
        #    src.agent.learn(encode(next), ops(action), -reward, encode(src.cur), False)
        # 反向q函数的思路不行
        #src.ragent.check_state_exist(encode(next))
        #src.ragent.learn(encode(next), ops(action), reward, encode(src.cur), done)

        src.steps = (1 + src.steps) % self.jumpgap
        src.cur = next
        self.move_block(src, next)

        # 碰撞（出界）或者满jumpgap步，随机跳
        if done or src.steps == 0: # 碰撞（出界），但没抵达终点。这些情况下随机跳
            if self.mode == 0:
                src.rjump()
            else:
                src.cur = src.start
            self.move_block(src, src.cur)

        return src

    def trymerge(self, src : Source, next):
        # 如果自己走过就不合并了
        if src.contain(next):
            return src
        target = self.intersection(next, src.name)
        # 别人也没走过
        if target is None:
            src.append(next)
            self.add_block(src, next)
            return src
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

        #print('{} {} merges {} at {}'.format(
        #    datetime.datetime.now(), eater.name, food.name, src.cur))
        eater.cur = src.cur
        eater.points.extend(food.points)
        eater.name = '{}to{}'.format(eater.start, eater.end)
        eater.isstart |= food.isstart
        eater.isend |= food.isend
        eater.steps = 0
        # 合并qtable
        eater.agent.q_table = self.merge_qtable(eater.agent.q_table, food.agent.q_table)

        self.srcs.remove(food)
        #self.display()
        #self.inner_train(eater)
        return eater

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
            guide_table(src.agent.q_table, self.height, self.width, '{}/{} ({})'.format(self.expname, length, src.name))

    def inner_train(self):
        self.finalsource.steps = 0
        self.finalsource.cur = self.start
        stime = datetime.datetime.now()
        while True:
            action = self.finalsource.agent.choose_action(encode(self.finalsource.cur))
            reward, done, info, next = super().step(self.finalsource, action)
            self.finalsource.agent.learn(encode(self.finalsource.cur), action, reward, encode(next), done)

            if info == FOUND:
                etime = datetime.datetime.now()
                return etime - stime
            self.finalsource.steps = (1 + self.finalsource.steps) % 500
            if done or self.finalsource.steps == 0:
                self.finalsource.cur = self.start
            else:
                self.finalsource.cur = next

'''跑完再写文档'''
def test(srcs, mode):
    data = []
    for _ in range(100):
        ms = MultipleReversal(srcs, mode)
        td = ms.explore()
        data.append(td.total_seconds())
    data1 = [round(x, 2) for x in data]
    fname = ename(mode, srcs)
    with open('../img/{}merge.txt'.format(fname), 'a', encoding='utf-8') as f:
        f.write('\'{}\':{}{}'.format(fname, str(data1), ',\n'))

'''有结果马上写文档里'''
def test1(srcs, mode):
    fname = ename(mode, srcs)
    for _ in range(100):
        ms = MultipleReversal(srcs, mode)
        td1 = ms.explore()
        td2 = ms.inner_train()
        with open('../img/{}merge.txt'.format(fname), 'a', encoding='utf-8') as f:
            f.write('{},\n'.format([td1.total_seconds(), td2.total_seconds()]))

srcdata = [
    [np.array([8, 2]),np.array([10, 7]),np.array([15, 14]),], # 0
    [np.array([8, 2]),np.array([15, 14]),], # 1
    [np.array([15, 14]),], # 2
    [np.array([13, 10]),], # 3
    [np.array([15, 7])], # 4
    [np.array([10, 7]),], # 5
    [np.array([8, 2]),], # 6
    [], #7
]

def ename(mode, srcs):
    if mode == 0 and len(srcs) == 0:
        return 'SE'
    elif mode == 0 and len(srcs) > 0:
        n = 'SPaSE{}'.format(len(srcs) + 1)
        for s in srcs:
            n += '_{}{}'.format(s[0], s[1])
        return n
    elif mode == 1 and len(srcs) == 0:
        return 'RFE'
    elif mode == 1 and len(srcs) > 0:
        n = 'SP{}'.format(len(srcs) + 1)
        for s in srcs:
            n += '_{}{}'.format(s[0], s[1])
        return n
    else:
        return 'QLearning'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, )
    parser.add_argument('--mode', type=int, default=0, help='0随机 1不随机')
    args = parser.parse_args()
    mode = args.mode
    n = args.n

    test1(srcdata[n], mode)
    #STEP_REWARD = -0.01
    #test1(srcdata[n], mode )
