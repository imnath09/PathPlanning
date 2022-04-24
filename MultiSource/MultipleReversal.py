
from MultiBase import *

class MultipleReversal(MultiBase):
    def __init__(self):
        super().__init__()
        self.agent = self.srcs[0].agent

    def walk(self, src : Source):
        action = src.agent.choose_action(encode(src.cur))
        next = src.cur + DIRECTION[action]
        target = None
        # 出界
        if (next[0] < 0 or
            next[1] < 0 or
            next[0] >= self.height or
            next[1] >= self.width):
            reward = CRASH_REWARD
            done = END_IF_OUT # 出界是否结束
            if not END_IF_OUT:
                next = src.cur # 出界结束就随机跳，否则回退
            info = OUT
        # 碰撞
        elif any((next == x).all() for x in self.obstacles):
            reward = CRASH_REWARD
            done = True
            info = CRASH
        # 抵达目的地
        elif (next == self.destination).all():
            reward = ARRIVE_REWARD
            #print('daoda{}'.format(reward))
            done = True
            info = ARRIVE
        # 没走过的点
        elif not src.contain(next):
            done = False
            info = WALK
            target = self.intersection(next, src.name)
            if target is not None: # 走进别人地盘
                reward = ARRIVE_REWARD # ????????????????????? 连接点是否给奖励需要斟酌??????????????
            else: # 没人走过的地方
                src.append(next)
                self.add_block(src, next)
                reward = STEP_REWARD
        # 移动在自己走过的路上
        else:
            reward = STEP_REWARD
            done = False
            info = WALK

        src.agent.learn(encode(src.cur), action, reward, encode(next), done)
        src.ragent.check_state_exist(encode(next))
        src.ragent.learn(encode(next), ops(action), reward, encode(src.cur), done)
        
        src.steps = (1 + src.steps) % 50
        src.cur = next
        self.move_block(src, next)

        if target is not None:
            info = self.merge(src, target)

        # 碰撞（出界）或者满50步，随机跳
        if done or src.steps == 0: # 碰撞（出界），但没抵达终点。这些情况下随机跳
            src.rjump()
            self.move_block(src, src.cur)

        return info

    def merge(self, walker : Source, other : Source):
        # 先确定谁吃谁
        walker_eats : bool
        if walker.isstart or other.isend:
            eater, food = walker, other
            walker_eats = True
        elif walker.isend or other.isstart:
            eater, food = other, walker
            walker_eats = False
        else:
            w = (walker.points[0][0] - self.start[0])**2 + (walker.points[0][1] - self.start[1])**2
            o = (other.points[0][0] - self.start[0])**2 + (other.points[0][1] - self.start[1])**2
            if w < o:
                eater, food = walker, other
                walker_eats = True
            else:
                eater, food = other, walker
                walker_eats = False
        print('{} merges {} at {}'.format(eater.name, food.name, walker.cur))

        eater.cur = eater.points[0]
        eater.points.extend(food.points)
        eater.isstart |= food.isstart
        eater.isend |= food.isend
        eater.steps = 0
        # 吃掉points

        # 合并qtable
        eater.agent.q_table = self.merge_qtable(eater.agent.q_table, food.agent.q_table)
        eater.ragent.q_table = self.merge_qtable(eater.ragent.q_table, food.ragent.q_table)

        self.inner_train(eater)

        self.srcs.remove(food)
        return FOUND if (eater.isstart and eater.isend) else MERGE

    def merge_qtable(self, eater, food):
        its = list(set(eater.index).intersection(set(food.index)))
        #print(its)
        if len(its) > 0:
            #print('intersection', its)
            food.drop(index = its, inplace = True)
        its = list(set(eater.index).intersection(set(food.index)))
        #print('after', its)
        return pd.concat([eater, food])

    def display(self):
        length = str(len(self.srcs))
        for src in self.srcs:
            guide_table(src.agent.q_table, self.height, self.width, length + src.name)

    def inner_train(self, src):
        pass

if __name__ == '__main__':
    ms = MultipleReversal()
    ms.explore()
