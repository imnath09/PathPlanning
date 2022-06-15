import matplotlib.pyplot as plt
import argparse
import os
import sys
sys.path.append('..')

from Maze.gymmaze import *
from MultiSource.RenderMap import *
from Algorithm.TorchDQN import DeepQNetwork
from Algorithm.QLearning import QLearningTable
from Algorithm.Sarsa import SarsaLambdaTable
from Common.utils import *

JUMP = 1

def Batch(isTrain, gap):
    path_len = [] # 记录成功时的路径长度
    for episode in range(1, gap + 1):
        observation = env.reset()

        if MODE == AgentType.Sarsa:
            action = agent.choose_action(encode(observation)) if isTrain else agent.action(encode(observation))
            agent.eligibility_trace *= 0
        # one trial
        for _ in range(500):
        #while True:
            if MODE == AgentType.Sarsa:
                observation_, reward, done, info = env.step(action)
                action_ = agent.choose_action(encode(observation)) if isTrain else agent.action(encode(observation))
                agent.learn(encode(observation), action, reward, encode(observation_), action_, done)
                observation = observation_
                action = action_
            else:
                action = agent.choose_action(encode(observation)) if isTrain else agent.action(encode(observation))
                observation_, reward, done, info = env.step(action)
                if isTrain:
                    learn(observation, action, reward, observation_, done)
                observation = observation_

            env.render()
            if done:
                if not isTrain:
                    endpoints[tuple(observation + [1, 1])] += 1

                if info == ARRIVE:
                    length = len(env.cur_path)
                    path_len.append(length)
                    if env.new_sln:
                        info = "path length:{}".format(length)
                        for n in env.cur_path:
                           info = ("{}->{}".format(info, str(n)))
                        print("epi-{}{} {}".format('t' if isTrain else 'f', episode, info))
                break
        # enf of while(one trial)
    # enf of for(trial process)
    return path_len

def learn(observation, action, reward, observation_, done):
    if MODE == AgentType.DQN:
        global JUMP
        agent.store_transition(observation, action, reward, observation_) # , done)
        if JUMP > 200 and JUMP % 5 == 0:# (DQN_step > 200): and (DQN_step % 5 == 0):实际没必要填充200次，也没必要隔五次训练
            agent.learn()
        JUMP += 1
    else:#QLearning
        agent.learn(encode(observation), action, reward, encode(observation_), done)

def core(test_gap, train_gap, total_iter):
    #Batch(isTrain=True, gap=10)#起始训练填充200次，起始没必要
    test_rate = [] # 成功率
    test_len = [] # 平均长度
    train_rate = []
    train_len = []

    stime = get_time()
    expname = '{} tr{}it{}ts{} {}'.format(stime, train_gap, total_iter, test_gap, MODE.name)
    os.makedirs('../img/{}'.format(expname))
    print(expname)

    train_info = '0 {} /\n'.format(get_time())

    for i in range(1, 1 + total_iter):
        train = Batch(isTrain = True, gap = train_gap)
        train_rate.append(len(train) / train_gap)
        arvlen = sum(train) / len(train) if len(train) > 0 else 0
        train_len.append(arvlen)

        test = Batch(isTrain = False, gap = test_gap)
        test_rate.append(len(test) / test_gap)
        arvlen = sum(test) / len(test) if len(test) > 0 else 0
        test_len.append(arvlen)

        if i % 10 == 0: # 只是为了看到进度的，有没有都行
            print('iter{} {}, {}'.format(i, get_time(), sum(test_rate) * test_gap))
        train_info = '{}{} {} {}\n'.format(train_info, i, get_time(), len(test) / test_gap)

    print(endpoints)
    if MODE == AgentType.DQN:
        plt.figure()
        plt.plot(np.arange(len(agent.cost_his)), agent.cost_his)
        plt.ylabel('cost')
        plt.xlabel('training steps')
        plt.savefig('../img/{}/cost.png'.format(expname))
        plt.close()
    else:
        guide_table(agent.q_table, env.height, env.width, '{}/guide'.format(expname), cmap='rainbow')
    with open('../img/{}/{}.txt'.format(expname, MODE.name), 'w', encoding='utf-8') as f:
        f.write(','.join([str(x) for x in test_rate]) + '\n')
        f.write(','.join([str(round(x, 2)) for x in test_len]) + '\n')
        f.write(','.join([str(x) for x in train_rate]) + '\n')
        f.write(','.join([str(round(x, 2)) for x in train_len]) + '\n')
        f.write(train_info)

def encode(pos):
    if MODE == AgentType.DQN:
        return pos
    r = '{},{}'.format(pos[0], pos[1])
    return r

def decode(index):
    if MODE == AgentType.DQN:
        return index
    l = index.split(',')
    for i in range(len(l)):
        l[i] = int(l[i])
    l = np.array(l)
    return l

if __name__ == '__main__':
    #analyz('../img/QLearningTable.txt', '../img/MSSE.txt')
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action = 'store_true', help = 'render or not')
    parser.add_argument('--testgap', type=int, default=10)
    parser.add_argument('--traingap', type=int, default=1000)
    parser.add_argument('--iter', type=int, default=100)
    parser.add_argument(
        '--mode', type=int, default=1, help = '0:DQN, 1:QLearning, 2:Sarsa, 3:MSSE')
    args = parser.parse_args()

    test_gap = args.testgap
    train_gap = args.traingap
    total_iter = args.iter
    rendered = args.render
    MODE = AgentType(args.mode)
    env = GymMaze() if rendered else UnrenderedMaze()
    if MODE == AgentType.QLearning:
        agent = QLearningTable(actions=env.action_space, e_greedy=0.9)
    elif MODE == AgentType.DQN:
        agent = DeepQNetwork(
            len(env.action_space), n_features = 2, memory_size = 2000, e_greedy = 0.9)
    elif MODE == AgentType.Sarsa:
        agent = SarsaLambdaTable(actions=env.action_space, e_greedy=0.9)
    elif MODE == AgentType.MSSE:
        msse = MultipleReversal()
        etime = msse.explore()
        print('total merging time', etime, )
        agent = msse.agent
    endpoints = np.zeros((env.height + 2, env.width + 2), dtype = int)
    core(test_gap, train_gap, total_iter)

