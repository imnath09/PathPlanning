
from math import exp
from os import stat_result
import sys

from gym.envs.classic_control.rendering import LineWidth
sys.path.append('..')

from Maze.unrenderedmaze import *
from Maze.gymmaze import GymMaze
from Algorithm.Sarsa import SarsaLambdaTable
from Algorithm.QLearning import QLearningTable
from Algorithm.TorchDQN import DeepQNetwork

from HZJ.dabeijing import *

import matplotlib.pyplot as plt
import argparse

from Common.dmdp_actions import *
import time
import gym

STAT_GAP = 10
TOTAL_EPISODE = 100 * STAT_GAP



def Test(mode):
    suc_matrix = np.array([[0, 0],[0, 0]]) # 幕抵达矩阵
    suc = [[], []] # 成功率

    t_len = np.array([0, 0]) # 平均长度
    avr_len = [[], []] # 

    x = []
    endpoints = np.zeros((env.height + 2, env.width + 2), dtype = int)

    suc_length = [[[], []], [[], []]] # 路径长度

    DQN_step = 0

    print(time.ctime())
    for episode in range(1, TOTAL_EPISODE + 1):
        observation = env.reset()
        explore = False # 探索

        if mode == 0: # sarsa
            action = RL.choose_action(encode(observation))
            RL.eligibility_trace *= 0

        while True:
            if mode == 0: # 0:sarsa
                observation_, reward, done, info = env.step(action)
                action_ = RL.choose_action(encode(observation_))
                RL.learn(encode(observation), action, reward, encode(observation_), action_, done)
                observation = observation_
                action = action_
            elif mode == 1: # 1:QLearning
                action = RL.choose_action(encode(observation))
                observation_, reward, done, info = env.step(action)
                RL.learn(encode(observation), action, reward, encode(observation_), done)
                observation = observation_
            else: # 2:DQN
                action = RL.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                RL.store_transition(observation, action, reward, observation_)#, done)
                if (DQN_step > 200) and (DQN_step % 5==0):
                    RL.learn()
                observation = observation_

            env.render()
            explore |= RL.random_action
            DQN_step += 1
            if done:
                if not explore:
                    endpoints[tuple(observation + [1, 1])] += 1
                i = 0 if explore else 1 # 0是探索，1是利用
                suc_matrix[i][0] += 1 # 总次数 探索/利用
                suc_length[i][0].append(episode)

                if info == ARRIVE: # 抵达目的地
                    suc_matrix[i][1] += 1 # 成功次数 探索/利用
                    cpath = env.cur_path # env.d #
                    suc_length[i][1].append(len(cpath)) # 路径长度 探索/利用
                    t_len[i] += len(cpath)

                    if env.new_sln:
                        info = "path length:{}".format(len(cpath))
                        for n in cpath:
                           info = ("{}->{}".format(info, str(n)))
                        print("episode{} {}".format(episode, info))
                else: # 没有抵达目的地
                    suc_length[i][1].append(0)
                break

        if episode % STAT_GAP == 0:
            # 探索成功率
            c0 = 0 if suc_matrix[0][1] == 0 else round(suc_matrix[0][1] / suc_matrix[0][0], 4)
            # 利用成功率
            c1 = 0 if suc_matrix[1][1] == 0 else round(suc_matrix[1][1] / suc_matrix[1][0], 4)
            suc[0].append(c0)
            suc[1].append(c1)

            al0 = 0 if t_len[0] == 0 else round(t_len[0] / suc_matrix[0][1], 4) # 探索平均长度
            al1 = 0 if t_len[1] == 0 else round(t_len[1] / suc_matrix[1][1], 4) # 利用平均长度
            avr_len[0].append(al0)
            avr_len[1].append(al1)

            x.append(episode)

            if True:
                print("eps:{}; explore:{}/{}={}, len:{}; exploit:{}/{}={}, len:{};".format(
                episode,
                suc_matrix[0][1], suc_matrix[0][0], c0, al0,
                suc_matrix[1][1], suc_matrix[1][0], c1, al1))

            suc_matrix *= 0
            t_len *= 0

    print(endpoints)
    print('game over', time.ctime())

    fig = plt.figure(figsize = (15,10))
    ax = fig.subplots(2, 2)
    # 成功率
    ax[0][0].plot(x, suc[0], 'r-', label = 'train rate')
    ax[0][0].plot(x, suc[1], 'b-', label = 'test rate')
    ax[0][0].set_ylabel('success rate')
    ax[0][0].set_xlabel('episode')
    ax[0][0].legend()

    # 平均路径长度
    ax[0][1].plot(x, avr_len[0], 'r-', label = 'train length')
    ax[0][1].plot(x, avr_len[1], 'b-', label = 'test length')
    ax[0][1].set_ylabel('average length')
    ax[0][1].set_xlabel('episode')
    ax[0][1].legend()

    # 寻路情况
    ax[1][0].plot(suc_length[0][0], suc_length[0][1], 'r.', linewidth = 0.10, label = 'train length')
    ax[1][0].plot(suc_length[1][0], suc_length[1][1], 'b.', linewidth = 0.10, label = 'test length')
    ax[1][0].set_ylabel('path length')
    ax[1][0].set_xlabel('episode')
    ax[1][0].legend()

    # 终点热力图
    #endpoints[3][3] /= 8
    ax[1][1].imshow(endpoints, cmap = 'gray')

    #plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
    #plt.rcParams['axes.unicode_minus']=False
    plt.tight_layout()
    #plt.show()
    plt.savefig('../img/{}.png'.format(get_time()))
    plt.close('all')
    if mode == 0 or mode ==1:
        show_table(RL.q_table)
    else:
        RL.plot_cost()

def get_time():
    return time.strftime('%m-%d %H.%M.%S', time.localtime())



def show_table(table):
    ntbl = np.full((env.height + 2, env.width + 2), actions.stop.name)
    for r in table.index:
        pos = decode(r)
        s = table.loc[r]
        c = np.random.choice(s[s==np.max(s)].index)
        content = actions(c).name.ljust(5, ' ')
        ntbl[tuple(pos + [1, 1])] = content
    print(ntbl)

def encode(pos):
    r = '{},{}'.format(pos[0], pos[1])
    return r

def decode(index):
    l = index.split(',')
    for i in range(len(l)):
        l[i] = int(l[i])
    l = np.array(l)
    return l

def encode1(pos):
    i = (pos + [1, 1])[0]
    j = (pos + [1, 1])[1]
    r = i * (env.width + 2) + j
    return r

def decode1(index):
    j = index % (env.width + 2)
    i = int(index / (env.width + 2))
    return np.array([i - 1, j- 1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action = 'store_true', help = 'render or not')
    parser.add_argument(
        '--mode', type = int,choices = [0, 1, 2],
        help = '0:sarsa, 1:q learning, 2:DQN')
    args = parser.parse_args()
    rendered = args.render
    mode = args.mode
    env = GymMaze() if rendered else UnrenderedMaze()
    #env = Environment()
    RL = SarsaLambdaTable(actions=env.action_space) if mode == 0 else (
        QLearningTable(actions=env.action_space) if mode == 1 else
        DeepQNetwork(len(env.action_space), n_features = 2, memory_size = 2000, e_greedy = 0.9)
        )
    print(type(RL))
    Test(mode)


