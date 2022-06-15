
import sys
sys.path.append('..')
import argparse
from MultiSource.MultiBase import *

class MultipleReversal(MultiBase):
    def __init__(self, sources = None, mode = 0):
        MultiBase.__init__(self, sources = sources)
        self.agent = self.srcs[0].agent
        self.mode = mode # 0:MSSE; 1:RFE; 

    def walk(self, src : Source):
        action = src.agent.choose_action(encode(src.cur))
        reward, done, info, next = super().step(src, action)

        if info == WALK:# 尝试合并
            src = self.trymerge(src, next)

        src.agent.learn(encode(src.cur), action, reward, encode(next), done)
        if reward > 0:
            src.agent.learn(encode(next), ops(action), -reward, encode(src.cur), False)
        # 反向q函数的思路不行
        #src.ragent.check_state_exist(encode(next))
        #src.ragent.learn(encode(next), ops(action), reward, encode(src.cur), done)

        src.steps = (1 + src.steps) % 100
        src.cur = next
        self.move_block(src, next)

        # 碰撞（出界）或者满50步，随机跳
        if self.mode == 0 and (done or src.steps == 0): # 碰撞（出界），但没抵达终点。这些情况下随机跳
            src.rjump()
            self.move_block(src, src.cur)
        if self.mode == 1 and done:# or src.steps == 0: # 碰撞（出界），但没抵达终点。这些情况下回到起点
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
        self.display()
        self.inner_train(eater)
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
            guide_table(src.agent.q_table, self.height, self.width, length + src.name)

    def inner_train(self, src):
        pass

def test(srcs, mode=0):
    data = []
    for _ in range(100):
        ms = MultipleReversal(srcs, mode)
        td = ms.explore()
        data.append(td.total_seconds())
    data1 = [round(x, 2) for x in data]
    fn = '_'.join(['{}{}'.format(x[0], x[1]) for x in srcs])
    with open('../img/{}_{}.txt'.format(mode, fn), 'a', encoding='utf-8') as f:
        f.write('{}={}{}'.format(fn, str(data1), ',\n'))

data = [
    [np.array([8, 2]),np.array([10, 7]),np.array([15, 14]),],
    [np.array([8, 2]),np.array([15, 14]),],
    [np.array([8, 2]),],
    [np.array([10, 7]),],
    [np.array([13, 10]),],
    [np.array([15, 14]),],
    [np.array([15, 7])],
    [],
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, )
    args = parser.parse_args()
    n = args.n

    test(data[n], mode = 0)
    STEP_REWARD = -0.01
    test(data[n], mode = 0)
