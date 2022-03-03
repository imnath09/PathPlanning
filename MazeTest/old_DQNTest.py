
import sys
sys.path.append('..')

from Maze.unrenderedmaze import *
from Maze.gymmaze import GymMaze
from Algorithm.TorchDQN import DeepQNetwork

import matplotlib.pyplot as plt
import argparse

from Common.dmdp_actions import *
import time

STAT_GAP = 1000
TOTAL_EPISODE = 50 * STAT_GAP


def DQNTest():
    suc_matrix = np.array([[0, 0],[0, 0]]) # 幕抵达矩阵
    suc = [[], []] # 成功率

    t_len = np.array([0, 0]) # 平均长度
    avr_len = [[], []] # 

    x = []
    endpoints = np.zeros((MAZE_HEIGHT + 2, MAZE_WIDTH + 2), dtype = int)

    suc_len = [[[], []], [[], []]] # 路径长度

    DQN_step = 0

    print(time.ctime())
    for episode in range(1, TOTAL_EPISODE + 1):
        observation = env.reset()
        explore = False # 探索

        # one trial
        while True:
            action = RL.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            RL.store_transition(observation, action, reward, observation_) # , done)
            if True: # (DQN_step > 500) and (DQN_step % 5 == 0):
                RL.learn()
            observation = observation_
            env.render()

            explore |= RL.random_action
            if done:
                endpoints[tuple(observation + [1, 1])] += 1
                i = 0 if explore else 1 # 0是探索，1是利用
                suc_matrix[i][0] += 1 # 总次数 探索/利用
                suc_len[i][0].append(episode)

                if reward == ARRIVE_REWARD: # 抵达目的地
                    suc_matrix[i][1] += 1 # 成功次数 探索/利用
                    suc_len[i][1].append(env.cur_path.__len__()) # 路径长度 探索/利用
                    t_len[i] += env.cur_path.__len__()

                    if env.new_sln:
                        info = "path length:{}".format(env.cur_path.__len__())
                        for n in env.cur_path:
                           info = ("{}->{}".format(info, str(n)))
                        print("episode{} {}".format(episode, info))
                else: # 没有抵达目的地
                    suc_len[i][1].append(0)
                break
            DQN_step += 1
        # enf of while(one trial)

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
    # enf of for(train process)
    display(endpoints, x, suc, avr_len, suc_len)


def display(endpoints, x, suc, avr_len, suc_len):
    print('game over', time.ctime())
    print(endpoints)
    RL.plot_cost()

    fig, ax = plt.subplots(2, 2)
    # 成功率
    ax[0][0].plot(x, suc[0], 'r-')
    ax[0][0].plot(x, suc[1], 'b-')

    # 平均路径长度
    ax[0][1].plot(x, avr_len[0], 'r-')
    ax[0][1].plot(x, avr_len[1], 'b-')

    # 寻路情况
    ax[1][0].plot(suc_len[0][0], suc_len[0][1], 'r.', linewidth = 0.10)
    ax[1][0].plot(suc_len[1][0], suc_len[1][1], 'b.', linewidth = 0.10)

    # 终点热力图
    #endpoints[3][3] /= 8
    ax[1][1].imshow(endpoints, cmap = 'gray')

    plt.tight_layout()
    plt.plot()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action = 'store_true', help = 'render or not')
    args = parser.parse_args()
    rendered = args.render

    env = GymMaze() if rendered else UnrenderedMaze()
    RL = DeepQNetwork(env.action_space.__len__(), n_features = 2)#, memory_size = 2000)
    DQNTest()


