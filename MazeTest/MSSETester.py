
import os
import sys
sys.path.append('..')

from MultiSource.MultipleReversal import *

class MSSETester():
    def __init__(self, mode : AgentType, sources = None):
        '''
        mode : 1-QLearning; 4-RFE; 3-MSSE
        '''
        self.expname = '{} tr{}it{}ts{} {}'.format(get_time(), train_gap, total_iter, test_gap, mode.name)
        self.mode = mode
        os.makedirs('../img/{}'.format(self.expname))
        print(self.expname, 'begin')

        self.env = UnrenderedMaze()
        if mode == AgentType.QLearning:
            self.agent = QLearningTable(actions=self.env.action_space, e_greedy=0.9)
        elif mode == AgentType.RFE:
            msse = MultipleReversal(sources = sources, mode = 1, expname = self.expname)
            etime = msse.explore()
            print('total merging time', etime)
            self.agent = msse.agent
        elif mode == AgentType.MSSE:
            msse = MultipleReversal(sources = sources, mode = 0, expname = self.expname)
            etime = msse.explore()
            print('total merging time', etime)
            self.agent = msse.agent
        self.endpoints = np.zeros((self.env.height + 2, self.env.width + 2), dtype = int)


    def core(self, test_gap, train_gap, total_iter):
        test_rate = [] # 成功率
        test_len = [] # 平均长度
        train_rate = []
        train_len = []

        self.train_info = '0 {} /\n'.format(get_time())

        for i in range(1, 1 + total_iter):
            train = self.Batch(isTrain = True, gap = train_gap)
            train_rate.append(len(train) / train_gap)
            arvlen = sum(train) / len(train) if len(train) > 0 else 0
            train_len.append(arvlen)

            test = self.Batch(isTrain = False, gap = test_gap)
            test_rate.append(len(test) / test_gap)
            arvlen = sum(test) / len(test) if len(test) > 0 else 0
            test_len.append(arvlen)

            if i % 10 == 0: # 只是为了看到进度的，有没有都行
                print('iter{} {}, {}'.format(i, get_time(), sum(test_rate) * test_gap))
            self.train_info = '{}{} {} {}\n'.format(self.train_info, i, get_time(), len(test) / test_gap)

        print(self.endpoints)
        guide_table(self.agent.q_table, self.env.height, self.env.width, '{}/guide'.format(self.expname), cmap='rainbow')
        with open('../img/{}/{}.txt'.format(self.expname, self.mode.name), 'w', encoding='utf-8') as f:
            f.write(','.join([str(x) for x in test_rate]) + '\n')
            f.write(','.join([str(round(x, 2)) for x in test_len]) + '\n')
            f.write(','.join([str(x) for x in train_rate]) + '\n')
            f.write(','.join([str(round(x, 2)) for x in train_len]) + '\n')
            f.write(self.train_info)

    def Batch(self, isTrain, gap):
        path_len = [] # 记录成功时的路径长度
        for episode in range(1, gap + 1):
            observation = self.env.reset()

            # one trial
            for _ in range(500):
            #while True:
                action = self.agent.choose_action(encode(observation)) if isTrain else self.agent.action(encode(observation))
                observation_, reward, done, info = self.env.step(action)
                if isTrain:
                    self.agent.learn(encode(observation), action, reward, encode(observation_), done)
                observation = observation_

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
        # enf of for(trial process)
        return path_len

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
    parser.add_argument('--testgap', type=int, default=10)
    parser.add_argument('--traingap', type=int, default=1000)
    parser.add_argument('--iter', type=int, default=100)
    parser.add_argument('--mode', type=int, help = 'mode:1-QLearning;4-RFE;3-MSSE')
    parser.add_argument('--n', type=int)
    args = parser.parse_args()

    test_gap = args.testgap
    train_gap = args.traingap
    total_iter = args.iter
    mode = AgentType(args.mode)
    n = args.n
    msse = MSSETester(mode=mode, sources = data[n])
    msse.core(test_gap, train_gap, total_iter)



