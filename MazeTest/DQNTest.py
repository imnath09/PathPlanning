
import sys

sys.path.append('..')

from Maze.unrenderedmaze import *
from Maze.gymmaze import GymMaze
from Algorithm.TorchDQN import DeepQNetwork

import matplotlib.pyplot as plt
import argparse

from Common.dmdp_actions import *
import time

TRAIN_GAP = 1000
TEST_GAP = 100
TOTAL_ITER = 100


def Batch(isTrain : bool):
    path_len = [] # 记录成功时的路径长度
    DQN_step = 1
    gap = TRAIN_GAP if isTrain else TEST_GAP
    for episode in range(1, gap + 1):
        observation = env.reset()
        # one trial
        #for _ in range(100):
        while True:
            action = RL.choose_action(observation) if isTrain else RL.action(observation)
            observation_, reward, done, info = env.step(action)
            if isTrain:
                RL.store_transition(observation, action, reward, observation_) # , done)
                if True: # (DQN_step > 500) and (DQN_step % 5 == 0):
                    RL.learn()
            observation = observation_
            env.render()

            if done:
                if reward == ARRIVE_REWARD:
                    lenth = len(env.cur_path)
                    path_len.append(lenth)
                    if env.new_sln:
                        info = "path length:{}".format(lenth)
                        for n in env.cur_path:
                           info = ("{}->{}".format(info, str(n)))
                        print("episode-{}{} {}".format('t' if isTrain else 'f', episode, info))
                break
            DQN_step += 1
        # enf of while(one trial)
    # enf of for(trial process)
    return path_len

def core():
    x = []
    test_rate = [] # 成功率
    test_len = [] # 平均长度
    train_rate = []
    train_len = []

    for i in range(TOTAL_ITER):
        train = Batch(isTrain = True)
        train_rate.append(len(train) / TRAIN_GAP)
        train_len.append(sum(train) / TRAIN_GAP)

        test = Batch(isTrain = False)
        print('iteration{}, {} successes'.format(i, len(test)))
        test_rate.append(len(test) / TEST_GAP)
        test_len.append(sum(test) / TEST_GAP)
        x.append(i)

    #RL.plot_cost()
    fig = plt.figure(figsize = (10, 10))
    ax = fig.subplots(2, 2)

    ax[0, 0].plot(x, test_rate, 'r')
    ax[0, 1].plot(x, test_len, 'r')

    ax[1, 0].plot(x, train_rate, 'b')
    ax[1, 1].plot(x, train_len, 'b')

    plt.tight_layout()
    plt.plot()
    plt.show()


def display_old(endpoints, x, suc, avr_len, suc_len):
    print('game over', time.ctime())
    print(endpoints)
    RL.plot_cost()

    fig, ax = plt.subplots(2)
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
    RL = DeepQNetwork(len(env.action_space), n_features = 2, memory_size = 2000, e_greedy = 0.5)
    print("begin at ", time.ctime())
    core()
    print("finish at ", time.ctime())




