
import sys
sys.path.append('..')

from Algs.RFE import *
from Algs.Walker import Walker

class SpacePartition(RFE):
    def __init__(self, agent, env, n_walkers=3):
        RFE.__init__(self, agent, env)
        self.init_walkers(n_walkers)

    def init_walkers(self, n):
        if n < 1:
            raise ValueError('number of walkers must be larger than 0!')
        self.walkers = [Walker(self.env.start, isstart=True)]
        while len(self.walkers) < n:
            i, j = np.random.randint(self.env.height), np.random.randint(self.env.width)
            #print(i, j, self.env.map[i, j])
            x = np.array([i, j])
            if self.env.map[i, j] == 0 and not \
                any(np.array_equal(x, w.start) for w in self.walkers):
                self.walkers.append(Walker(x))
        info = ' + '.join(x.name for x in self.walkers)
        print(info)

    # UAVs.train(self):

    # UAVs.run_episode(self):

    def explore(self, horizon):
        fnl = None
        stime = datetime.datetime.now()
        datas = []
        while fnl is None:
            for walker in self.walkers:
                action = self.agent.choose_action(walker.cur)
                next, reward, done, info = self.env.step(action)
                datas.append((walker.cur, action, reward, next, done))
                if len(datas) > 100:
                    self.agent.store_transition_batch(datas)#sln2 
                    self.agent.learn_batch(datas)#sln2
                    datas = []

                bMerge = False
                if info == WALK:
                    bMerge = self.trymerge(walker, next)
                elif info == ARRIVE:
                    walker.isend = True

                walker.steps = walker.steps + 1
                walker.cur = next

                if done or walker.steps > horizon:
                    walker.jump()

                self.env.render()

                if walker.isstart and walker.isend:
                    fnl = walker
                    break
                elif bMerge:
                    break

        etime = datetime.datetime.now()
        return (etime - stime).total_seconds()

    def intersection(self, pos, walker):
        for w in self.walkers:
            if not w.name == walker.name and w.contain(pos):
                return w
        return None

    def trymerge(self, walker:Walker, next):
        # 如果自己走过就不合并了
        if walker.contain(next):
            return False

        target = self.intersection(next, walker)
        # 别人也没走过
        if target is None:
            walker.append(next)
            #self.add_block(walker, next)
            return False

        # 如果走进别人的地盘
        walker.points.extend(target.points)
        walker.name = '{}to{}'.format(walker.name, target.name)
        walker.isstart |= target.isstart
        walker.isend |= target.isend
        walker.steps = 0 #+= target.steps
        self.walkers.remove(target)

        return True





