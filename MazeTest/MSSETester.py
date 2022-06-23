
import os
import sys
sys.path.append('..')

from MultiSource.MultipleReversal import *

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

class MSSETester():
    def __init__(self, mode, sources):
        fname = ename(mode, sources)
        self.expname = '{} tr{}it{}ts{} {}'.format(get_time(), train_gap, total_iter, test_gap, fname, )
        self.mode = mode
        self.sources = sources
        os.makedirs('../img/{}'.format(self.expname))
        print(self.expname, 'begin')

        self.env = UnrenderedMaze()
        if mode == 2:
            self.agent = QLearningTable(actions=self.env.action_space, e_greedy=0.9)
        else:
            msse = MultipleReversal(sources = sources, mode = mode, expname = self.expname)
            self.mergetime = msse.explore()
            self.exploretime = msse.inner_train()
            self.agent = msse.agent

        self.endpoints = np.zeros((self.env.height + 2, self.env.width + 2), dtype = int)


    def core(self, test_gap, train_gap, total_iter):
        test_rate = [] # 成功率
        test_len = [] # 平均长度
        test_reward = []
        train_rate = []
        train_len = []
        train_reward = []

        self.train_info = '0 {} /\n'.format(get_time())
        stime = datetime.datetime.now()
        cvgtime = None

        for i in range(1, 1 + total_iter):
            train, re = self.Batch(isTrain = True, gap = train_gap)
            train_rate.append(len(train) / train_gap)
            arvlen = sum(train) / len(train) if len(train) > 0 else 0
            train_len.append(arvlen)
            train_reward.append(sum(re) / len(re))

            test, re = self.Batch(isTrain = False, gap = test_gap)
            trate = len(test) / test_gap
            test_rate.append(trate)
            if trate < 1.0:
                cvgtime = None
            elif cvgtime is None:
                cvgtime = datetime.datetime.now()
            arvlen = sum(test) / len(test) if len(test) > 0 else 0
            test_len.append(arvlen)
            test_reward.append(sum(re) / len(re))

            #if i % 10 == 0: # 只是为了看到进度的，有没有都行
            #    print('iter{} {}, {}'.format(i, get_time(), sum(test_rate) * test_gap))
            self.train_info = '{}{} {} {}\n'.format(self.train_info, i, get_time(), trate)
        # end of for
        plntime = datetime.datetime.now() - stime
        cvg = (cvgtime - stime) if cvgtime is not None else (stime - stime)
        train_time = [self.mergetime.total_seconds(),self.exploretime.total_seconds(),cvg.total_seconds(),plntime.total_seconds()] if hasattr(self, 'mergetime') else [cvg.total_seconds(),plntime.total_seconds()]
        #print(self.endpoints)
        guide_table(self.agent.q_table, self.env.height, self.env.width, '{}/guide'.format(self.expname), cmap='rainbow')
        with open('../img/{}/data.txt'.format(self.expname), 'w', encoding='utf-8') as f:
            f.write(','.join([str(x) for x in test_rate]) + '\n')
            f.write(','.join([str(round(x, 2)) for x in test_len]) + '\n')
            f.write(','.join([str(x) for x in train_rate]) + '\n')
            f.write(','.join([str(round(x, 2)) for x in train_len]) + '\n')
            f.write(','.join([str(round(x, 3)) for x in test_reward]) + '\n')
            f.write(','.join([str(round(x, 3)) for x in train_reward]) + '\n')
            f.write('train time: {}\n'.format(train_time))
            f.write(self.train_info)
        with open('../img/{}.txt'.format(ename(self.mode, self.sources)), 'a', encoding='utf-8') as f:
            f.write('{},\n'.format(train_time))

    def Batch(self, isTrain, gap):
        path_len = [] # 记录成功时的路径长度
        accum_reward = [] # 记录累计奖励
        for episode in range(1, gap + 1):
            observation = self.env.reset()
            total_reward = 0

            # one trial
            for _ in range(500):
            #while True:
                action = self.agent.choose_action(encode(observation)) if isTrain else self.agent.action(encode(observation))
                observation_, reward, done, info = self.env.step(action)
                if isTrain:
                    self.agent.learn(encode(observation), action, reward, encode(observation_), done)
                observation = observation_
                total_reward += reward

                self.env.render()
                if done:
                    if not isTrain:
                        self.endpoints[tuple(observation + [1, 1])] += 1

                    if info == ARRIVE:
                        length = len(self.env.cur_path)
                        path_len.append(length)
                        if self.env.new_sln:
                            self.train_info += "epi-{}{} path:{} {}\n".format('t' if isTrain else 'f', episode, length, '-'.join([str(x) for x in self.env.cur_path]))
                    break
            # enf of while(one trial)
            accum_reward.append(total_reward)
        # enf of for(trial process)
        return path_len, accum_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testgap', type=int, default=10)
    parser.add_argument('--traingap', type=int, default=1000)
    parser.add_argument('--iter', type=int, default=100)
    parser.add_argument('--mode', type=int, help = 'mode:0-随机;1-不随机;2-qlearning')
    parser.add_argument('--n', type=int, default=0)
    args = parser.parse_args()

    test_gap = args.testgap
    train_gap = args.traingap
    total_iter = args.iter
    mode = args.mode
    n = args.n
    msse = MSSETester(mode=mode, sources = srcdata[n])
    msse.core(test_gap, train_gap, total_iter)



