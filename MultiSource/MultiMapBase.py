
from MultiBase import *

class MultiMapBase(MultiBase):
    def __init__(self):
        super().__init__()
        self.agent = QLearningTable(actions = range(4), e_greedy = 0.9)

    def walk(self, src : Source):
        action = self.agent.choose_action(encode(src.cur))
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

        self.agent.learn(encode(src.cur), action, reward, encode(next), done)
        self.agent.learn(encode(next), ops(action), -reward, encode(src.cur), done)

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

    # 走到别人地盘的被合并
    def merge(self, walker : Source, other : Source):
        other.points.extend(walker.points)
        other.isstart |= walker.isstart
        other.isend |= walker.isend
        other.steps = 0
        self.srcs.remove(walker)
        print('merge {} and {} at {}'.format(other.name, walker.name, walker.cur))
        return FOUND if (other.isstart and other.isend) else MERGE

if __name__ == '__main__':
    ms = MultiMapBase()
    ms.explore()
