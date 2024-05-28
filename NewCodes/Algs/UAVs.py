import sys
sys.path.append('..')

import datetime
import time

from MAPs.GymMap import *

class UAVs():
    def __init__(self, agent, env):
        self.env = env#map=np.zeros((10, 10))) # GymMaze(map=np.zeros((10, 10))) # 
        self.agent = agent
        print(self.__class__.__name__, 
              self.agent.__class__.__name__,
              self.env.__class__.__name__)
        # QLearningTable() # VallinaDQN() # ReplayDQN() # PriorDQN()

    def Planning(self, test_gap, train_gap, total_iter, horizon):
        test_rate = [] # 成功率
        test_len = [] # 平均长度
        test_reward = []
        train_rate = []
        train_len = []
        train_reward = []

        stime = datetime.datetime.now()
        # 进行total_iter次迭代
        for i in range(1, 1 + total_iter):
            # 先训练
            avg_len, avg_suc, avg_reward = self.run_episodes(isTrain=True, run_size=train_gap, horizon=horizon)
            train_len += avg_len
            train_rate += avg_suc
            train_reward += avg_reward

            # 再评估
            avg_len, avg_suc, avg_reward = self.run_episodes(isTrain=False, run_size=test_gap, horizon=horizon)
            test_rate.append(sum(avg_suc) / len(avg_suc))
            test_len.append(sum(avg_len) / len(avg_len))
            test_reward.append(sum(avg_reward) / len(avg_reward))

            if i % 10 == 0: # 只是为了看到进度的，有没有都行
                print('iter{} {}, {}'.format(i, get_time(), sum(avg_suc) / len(avg_suc)))
        # end of for
        etime = datetime.datetime.now()
        plan_time = (etime - stime).total_seconds()
        print('total planning time', plan_time)
        return plan_time


    def run_episodes(self, isTrain, run_size, horizon):
        path_len, suc, accum_reward = [], [], []
        for episode in range(1, run_size + 1):
            state = self.env.reset()
            total_reward = 0
            datas = []
            flag = 0
            # one episode
            for i in range(horizon):
                if isTrain:
                    action = self.agent.choose_action(state)
                else:
                    action = self.agent.action(state)
                    #print(action)
                state_, reward, done, info = self.env.step(action)
                if isTrain:
                    datas.append((state, action, reward, state_, done))
                    #self.agent.store_transition(state, action, reward, state_, done)
                    #self.agent.learn(state, action, reward, state_, done)
                state = state_
                total_reward += reward

                self.env.render()
                if done:
                    if info == ARRIVE:
                        flag = 1
                        if self.env.new_sln:
                            print("{} path-len:{} {}\n".format(
                                'train' if isTrain else 'test', 
                                i + 1, 
                                '-'.join([str(x) for x in self.env.cur_path])))
                    break
            # enf of (one episode)
            path_len.append(i + 1)
            accum_reward.append(total_reward)
            suc.append(flag)
            if isTrain:
                self.agent.store_transition_batch(datas)
                self.agent.learn_batch(datas)
        # end of (all episodes)
        return path_len, suc, accum_reward

    def explore(self, *args):
        # do nothing
        return 0


def get_time():
    return time.strftime('%m-%d %H.%M.%S', time.localtime())

