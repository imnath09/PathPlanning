
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
TEST_GAP = 10
TOTAL_ITER = 10000


def Batch(isTrain : bool, gap = 200, flag = True):
    path_len = [] # 记录成功时的路径长度
    DQN_step = 1
    for episode in range(1, gap + 1):
        observation = env.reset()
        # one trial
        for _ in range(500):
        #while True:
            action = RL.choose_action(observation) if isTrain else RL.action(observation)
            observation_, reward, done, info = env.step(action)
            if isTrain:
                RL.store_transition(observation, action, reward, observation_) # , done)
                if flag or DQN_step % 5 == 0:# (DQN_step > 200): and (DQN_step % 5 == 0):实际没必要填充200次，也没必要隔五次训练
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
    #Batch(isTrain=True, flag = False)#起始训练填充200次，起始没必要

    x = []
    test_rate = [] # 成功率
    test_len = [] # 平均长度
    train_rate = []
    train_len = []
    print("begin at ", time.ctime())
    for i in range(TOTAL_ITER):
        train = Batch(isTrain = True, gap = TRAIN_GAP, flag = False)
        train_rate.append(len(train) / TRAIN_GAP)
        train_len.append(sum(train) / TRAIN_GAP)

        test = Batch(isTrain = False, gap = TEST_GAP, flag = False)
        if len(test) < TEST_GAP:
            print('iteration{} {}, {} successes'.format(i, time.ctime(), len(test)))
        test_rate.append(len(test) / TEST_GAP)
        test_len.append(sum(test) / TEST_GAP)
        x.append(i)
    print("finish at ", time.ctime())

    fig = plt.figure(figsize = (10, 10))
    fig.suptitle('title')
    ax = fig.subplots(2, 2)

    ax[0, 0].plot(x, test_rate, 'r-')
    ax[0, 0].set_ylabel('test success rate')
    ax[0, 0].set_xlabel('training iteration')

    ax[0, 1].plot(x, test_len, 'r.')
    ax[0, 1].set_ylabel('average lenth')
    ax[0, 1].set_xlabel('training iteration')

    ax[1, 0].plot(x, train_rate, 'b')
    ax[1, 0].set_ylabel('test success rate')
    ax[1, 0].set_xlabel('training iteration')

    ax[1, 1].plot(np.arange(len(RL.cost_his)), RL.cost_his)
    ax[1, 1].set_ylabel('Cost')
    ax[1, 1].set_xlabel('training steps')

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
    core()



