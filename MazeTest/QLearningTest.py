
import sys
sys.path.append('..')

from Algorithm.QLearning import QLearningTable
from Maze.gymmaze import *

import matplotlib.pyplot as plt
import argparse

import time

def Batch(isTrain, gap, flag):
    path_len = [] # 记录成功时的路径长度
    DQN_step = 1
    for episode in range(1, gap + 1):
        observation = env.reset()
        # one trial
        for _ in range(500):
        #while True:
            action = RL.choose_action(encode(observation)) if isTrain else RL.action(encode(observation))
            observation_, reward, done, info = env.step(action)
            RL.learn(encode(observation), action, reward, encode(observation_), done)
            observation = observation_
            env.render()
            DQN_step += 1

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
        # enf of while(one trial)
    # enf of for(trial process)
    return path_len

def core(test_gap, train_gap, total_iter, cons):
    #Batch(isTrain=True, gap=200, flag = False)#起始训练填充200次，起始没必要
    x = []
    test_rate = [] # 成功率
    test_len = [] # 平均长度
    train_rate = []
    train_len = []
    starttime = get_time()
    count = 0
    print('start', get_time())
    for i in range(1, 1 + total_iter):
        train = Batch(isTrain = True, gap = train_gap, flag = cons)
        train_rate.append(len(train) / train_gap)
        train_len.append(sum(train) / train_gap)

        test = Batch(isTrain = False, gap = test_gap, flag = cons)
        if len(test) < test_gap:
            count += 1
            #print('iteration{} {}, {} successes'.format(i, get_time(), len(test)))
        if i % 10 == 0:
            print('iter{} {}, {} failed'.format(i, get_time(), count))
            count = 0
        test_rate.append(len(test) / test_gap)
        test_len.append(sum(test) / test_gap)
        x.append(i)
    endtime = get_time()
    info = 'from {} to {} testgap{} train{} iter{} cons{}'.format(
        starttime, endtime, test_gap, train_gap, total_iter, cons)
    print(info)

    fig = plt.figure(figsize = (15, 10))
    fig.suptitle(info)
    ax = fig.subplots(2, 2)

    ax[0, 0].plot(x, test_rate, 'r-')
    ax[0, 0].set_ylabel('test success rate')
    ax[0, 0].set_xlabel('test iteration')

    ax[0, 1].plot(x, test_len, 'r-')
    ax[0, 1].set_ylabel('average length')
    ax[0, 1].set_xlabel('test iteration')

    ax[1, 0].plot(x, train_rate, 'b')
    ax[1, 0].set_ylabel('train success rate')
    ax[1, 0].set_xlabel('training iteration')

    #ax[1, 1].plot(np.arange(len(RL.cost_his)), RL.cost_his)
    ax[1, 1].plot(x, train_len, 'b')
    ax[1, 1].set_ylabel('average length')
    ax[1, 1].set_xlabel('training steps')

    plt.tight_layout()
    plt.plot()
    plt.savefig('../img/{}.png'.format(get_time()))
    #plt.show()
    plt.close('all')

def get_time():
    return time.strftime('%m-%d %H.%M.%S', time.localtime())

def encode(pos):
    r = '{},{}'.format(pos[0], pos[1])
    return r

def decode(index):
    l = index.split(',')
    for i in range(len(l)):
        l[i] = int(l[i])
    l = np.array(l)
    return l

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action = 'store_true', help = 'render or not')
    parser.add_argument('--testgap', type=int, default=10)
    parser.add_argument('--traingap', type=int, default=100)
    parser.add_argument('--iter', type=int, default=500)
    parser.add_argument('--cons', type=bool, default=True)
    args = parser.parse_args()

    test_gap = args.testgap
    train_gap = args.traingap
    total_iter = args.iter
    cons = args.cons
    rendered = args.render
    env = GymMaze() if rendered else UnrenderedMaze()
    RL = QLearningTable(actions=env.action_space)
    core(test_gap, train_gap, total_iter, cons)


