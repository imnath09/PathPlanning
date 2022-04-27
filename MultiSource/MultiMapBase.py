
import sys
sys.path.append('..')
from MultiSource.MultiBase import *

class MultiMapBase(MultiBase):
    def __init__(self):
        MultiBase.__init__(self)
        self.agent = QLearningTable(actions = range(4), e_greedy = 0.9)

    def walk(self, src : Source):
        action = self.agent.choose_action(encode(src.cur))
        reward, done, info, next, target = self.step(src, action)

        self.agent.learn(encode(src.cur), action, reward, encode(next), done)
        #self.agent.learn(encode(next), ops(action), -reward, encode(src.cur), done)

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
