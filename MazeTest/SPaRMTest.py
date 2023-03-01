
import os
import sys
sys.path.append('..')

from MultiSource.SpacePartition import *
from MultiSource.SPaRM import *
from MultiSource.RFE import *

'''
1.qlearning，Qlearning
2.RFE，srcs=[],mode=1
3.SE，srcs=[],mode=0
4.SP，srcs=[...],mode=1
5.SPaSE，srcs=[...],mode=0

空间划分消融:space partition
1.一个源，500步后回到起点，直接规划终点——qlearning
2.一个源，500步后回到起点，先合并，再规划终点——RFE
4.多个源，500步后回到起点，先合并，再规划终点——space partition

随机探索消融:stochastic exploration
1.一个源，500步后回到起点，直接规划终点——qlearning
2.一个源，500步后回到起点，先合并，再规划终点——RFE
3.一个源，100步后随机跳转，先合并，再规划终点——stochastic exploration
'''

class SPaRMTest():
    def __init__(self, mode, sources):
        ep = SPaRMname(mode, sources)
        self.expname = '{} {}_{}_{} {}'.format(ep, total_iter, train_gap, test_gap, get_time())
        #os.makedirs('../img/{}'.format(self.expname))
        print(self.expname, 'begin')

        self.env = UnrenderedMaze()
        if mode == 2:
            self.wallclock_expand = 0
            self.wallclock_inner = 0
            self.episodes_expand = 0
            self.episodes_inner = 0
            self.agent = QLearningTable(actions=self.env.action_space, e_greedy=0.9)
        else:
            if mode == 1:
                msse = SpacePartition(sources = sources, expname = self.expname)
            elif mode == 0:
                msse = SPaRM(sources = sources, expname = self.expname)
            elif mode == 3:
                msse = RFE(expname = self.expname)

            msse.Exploration()
            self.agent = msse.finalsource.agent

            self.wallclock_expand = msse.wallclock_expand
            self.wallclock_inner = msse.wallclock_inner
            self.episodes_expand = msse.episodes_expand
            self.episodes_inner = msse.episodes_inner

    def Planning(self, test_gap, train_gap, total_iter):
        test_rate = [] # 成功率
        test_len = [] # 平均长度
        test_reward = []
        train_rate = []
        train_len = []
        train_reward = []

        cvgtime = None
        cvgiter = 0
        stime = datetime.datetime.now()
        # 进行total_iter次迭代
        for i in range(1, 1 + total_iter):
            # 先训练train_gap个episodes
            avg_len, avg_succ, avg_reward = self.Batch(isTrain = True, gap = train_gap)
            train_rate.append(avg_succ)
            train_len.append(avg_len)
            train_reward.append(avg_reward)
            # 再评估test_gap个episodes
            avg_len, avg_suc, avg_reward = self.Batch(isTrain = False, gap = test_gap)
            test_rate.append(avg_suc)
            test_len.append(avg_len)
            test_reward.append(avg_reward)
            if avg_suc < 1.0:
                cvgtime = None
                cvgiter = 0
            elif cvgtime is None:
                cvgtime = datetime.datetime.now()
                cvgiter = i

            #if i % 10 == 0: # 只是为了看到进度的，有没有都行
            #    print('iter{} {}, {}'.format(i, get_time(), avg_suc))
        # end of for
        convergence_time = (cvgtime - stime).total_seconds() if cvgtime is not None else 0
        etime = datetime.datetime.now()
        plan_time = (etime - stime).total_seconds()
        #guide_table(self.agent.q_table, self.env.height, self.env.width, '{}/guide'.format(self.expname), cmap='rainbow')
        with open('../img/{}.txt'.format(self.expname), 'w', encoding='utf-8') as f:
            #暂时不记录长度，但是代码要保留
            #f.write(','.join([str(round(x, 2)) for x in test_len]) + '\n')
            #f.write(','.join([str(round(x, 2)) for x in train_len]) + '\n')
            f.write(','.join([str(x) for x in test_rate]) + '\n')
            f.write(','.join([str(x) for x in train_rate]) + '\n')
            f.write(','.join([str(round(x, 3)) for x in test_reward]) + '\n')
            f.write(','.join([str(round(x, 3)) for x in train_reward]) + '\n')
            #f.write('{},{},{}\n'.format(
            f.write('{},{},{},{},{},{}\n'.format(
                self.wallclock_expand,
                self.wallclock_inner,
                convergence_time,
                self.wallclock_expand + self.wallclock_inner,
                self.wallclock_expand + self.wallclock_inner + convergence_time,
                self.wallclock_expand + self.wallclock_inner + plan_time,
                ))
            f.write('{},{},{},{},{}\n'.format(
                self.episodes_expand,
                self.episodes_inner,
                cvgiter * train_gap,
                self.episodes_expand + self.episodes_inner,
                self.episodes_expand + self.episodes_inner + cvgiter * train_gap,
            ))
        print(etime, 'end')

    def Batch(self, isTrain, gap):
        path_len = [] # 记录成功时的路径长度
        accum_reward = [] # 记录累计奖励
        for episode in range(1, gap + 1):
            observation = self.env.reset()
            total_reward = 0
            # one episode
            for i in range(PLANNING_HORIZON):
                action = self.agent.choose_action(encode(observation)) if isTrain else self.agent.action(encode(observation))
                observation_, reward, done, info = self.env.step(action)
                if isTrain:
                    self.agent.learn(encode(observation), action, reward, encode(observation_), done)
                observation = observation_
                total_reward += reward

                self.env.render()
                if done:
                    if info == ARRIVE:
                        length = len(self.env.cur_path)
                        path_len.append(length)
                        #if self.env.new_sln:
                        #    self.train_info += "{}-episode{} path:{} {}\n".format('train' if isTrain else 'test', i, length, '-'.join([str(x) for x in self.env.cur_path]))
                    break
            # enf of (one episode)
            accum_reward.append(total_reward)
        # enf of (one batch)
        avg_len = sum(path_len) / len(path_len) if len(path_len) > 0 else 0
        avg_suc = len(path_len) / gap
        avg_reward = sum(accum_reward) / gap
        return avg_len, avg_suc, avg_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testgap', type=int, default=3)
    parser.add_argument('--traingap', type=int, default=200)
    parser.add_argument('--iter', type=int, default=300)
    parser.add_argument('--mode', type=int, default=0,
                        help = '0-SPaRM; mode:1-SpacePartition;\
                            2-qlearning; 3-RFE')
    parser.add_argument('--n', type=int, default=1)
    args = parser.parse_args()

    test_gap = args.testgap
    train_gap = args.traingap
    total_iter = args.iter
    mode = args.mode
    n = args.n
    msse = SPaRMTest(mode=mode, sources = srcdata[n])
    msse.Planning(test_gap, train_gap, total_iter)






