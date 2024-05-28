
import sys
sys.path.append('..')

from Algs.UAVs import *

class RFE(UAVs):
    def __init__(self, agent, env):
        UAVs.__init__(self, agent, env)

    # UAVs.Planning(self):

    # UAVs.run_episodes(self):

    def explore(self, horizon):
        stime = datetime.datetime.now()
        i = 1
        j = 0 # sln1, sln2
        while True:
            state = self.env.reset()
            total_reward = 0
            datas = [] # sln1, sln2
            flag = 0
            for _ in range(horizon):
                action = self.agent.choose_action(state)
                next, reward, done, info = self.env.step(action)
                datas.append((state, action, reward, next, done))# sln1, sln2
                state = next
                total_reward += reward

                if done:
                    #在哪一级储存和训练尤为关键，这里是只学习done的事件。#sln1
                    #self.agent.store_transition_batch(datas) #sln1
                    #self.agent.learn_batch(datas) #sln1
                    if info == ARRIVE:
                        flag = 1
                        if self.env.new_sln:
                            print("path-len:{} {}\n".format(
                                len(self.env.s_path),
                                '-'.join([str(x) for x in self.env.cur_path])))
                    break
            #这里是学习所有事件 sln2
            self.agent.store_transition_batch(datas)#sln2 
            self.agent.learn_batch(datas)#sln2
            if i % 10000 == 0:
                print('epl for', i)
            i += 1
            j += len(datas)
            if flag:
                break
        print('explore for {} episodes. len of datas {}'.format(i - 1, j))
        #等到访问到终点才学习的话，没有学到经验，很难抵达终点。
        #self.agent.store_transition_batch(datas) #sln3
        #self.agent.learn_batch(datas) #sln3
        etime = datetime.datetime.now()
        epl_time = (etime - stime).total_seconds()
        print('total exploration time', epl_time)
        return epl_time
