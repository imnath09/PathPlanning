
from MultiSource.MultiBase import *

class MultipleReversal(MultiBase):
    def __init__(self):
        MultiBase.__init__(self)
        self.agent = self.srcs[0].agent

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
        if done or src.steps == 0: # 碰撞（出界），但没抵达终点。这些情况下随机跳
            src.rjump()
            #src.cur = src.start
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

        print('{} {} merges {} at {}'.format(
            datetime.datetime.now(), eater.name, food.name, src.cur))
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

if __name__ == '__main__':
    ms = MultipleReversal()
    ms.explore()
