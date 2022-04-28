import matplotlib.pyplot as plt
import argparse
import time

import sys
sys.path.append('..')

from Maze.gymmaze import *
from MultiSource.RenderMap import *
from Algorithm.TorchDQN import DeepQNetwork
from Algorithm.QLearning import QLearningTable

JUMP = 1

def Batch(isTrain, gap):
    path_len = [] # 记录成功时的路径长度
    for episode in range(1, gap + 1):
        observation = env.reset()
        # one trial
        for _ in range(2000):
        #while True:
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
                        print("episode-{}{} {}".format('t' if isTrain else 'f', episode, info))
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
    starttime = get_time()
    print('start', starttime)
    for i in range(1, 1 + total_iter):
        train = Batch(isTrain = True, gap = train_gap)
        train_rate.append(len(train) / train_gap)
        arvlen = sum(train) / len(train) if len(train) > 0 else 0
        train_len.append(arvlen)

        test = Batch(isTrain = False, gap = test_gap)
        test_rate.append(len(test) / test_gap)
        arvlen = sum(test) / len(test) if len(test) > 0 else 0
        test_len.append(arvlen)

        if i % 1 == 0:
            print('iter{} {}, {}'.format(i, get_time(), sum(test_rate) * test_gap))

    info = 'Cross{}to{}test{}train{}iter{}'.format(
        starttime, get_time(), test_gap, train_gap, total_iter)
    print(info)
    print(endpoints)
    display(info, test_rate, train_rate, test_len, train_len, '-')
    display(info, test_rate, train_rate, test_len, train_len, '.')

def display(info, test_rate, train_rate, test_len, train_len, stl):
    x = range(1, 1 + total_iter)
    fig = plt.figure(figsize = (15, 10))
    fig.suptitle(info)
    ax = fig.subplots(2, 2)
    # 成功率
    ax[0, 0].plot(x, test_rate, 'r'+stl, label = 'test rate')
    ax[0, 0].plot(x, train_rate, 'b'+stl, label = 'train rate')
    '''avrate = []
    for i in range(0, total_iter, 10):
        avrate.append(sum(test_rate[i: i + 9]) / 10)
    ax[0, 0].plot(range(10, total_iter + 1, 10), avrate, 'y'+stl, label = 'avrtest rate')'''
    ax[0, 0].set_ylabel('success rate,')
    ax[0, 0].set_xlabel('horizon')
    ax[0, 0].legend()
    # 平均长度
    ax[0, 1].plot(x, train_len, 'b'+stl, label = 'train length')
    ax[0, 1].plot(x, test_len, 'r'+stl, label = 'test length')
    ax[0, 1].set_ylabel('average length')
    ax[0, 1].set_xlabel('horizon')
    ax[0, 1].legend()
    # 终点热力图
    ax[1, 0].imshow(endpoints, cmap = 'gray')
    # 损失函数
    if MODE == AgentType.DQN:
        ax[1, 1].plot(np.arange(len(agent.cost_his)), agent.cost_his)
        ax[1, 1].set_ylabel('Cost')
        ax[1, 1].set_xlabel('training steps')
        #ax[1, 1].legend()
    # 方向导图
    else:
        guide_table(agent.q_table)

    #plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
    #plt.rcParams['axes.unicode_minus']=False
    plt.tight_layout()
    plt.savefig('../img/{}{}.png'.format(get_time(), stl))
    #plt.show()
    plt.close('all')

def get_time():
    return time.strftime('%m-%d %H.%M.%S', time.localtime())

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

def guide_table(table):
    ntbl = np.full((env.height + 2, env.width + 2), 4.0)
    for r in table.index:
        pos = tuple(decode(r) + [1, 1])
        s = table.loc[r]
        c = np.random.choice(s[s==np.max(s)].index)
        content = actions(c).name[0]#.ljust(5, ' ')
        ntbl[pos] = c
        plt.annotate(text=content, xy=(pos[1], pos[0]), ha='center', va='center')
    plt.imshow(ntbl, cmap = 'rainbow', vmin = 0, vmax = 4)
    plt.colorbar()
    return ntbl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action = 'store_true', help = 'render or not')
    parser.add_argument('--testgap', type=int, default=10)
    parser.add_argument('--traingap', type=int, default=1000)
    parser.add_argument('--iter', type=int, default=100)
    parser.add_argument('--mode', type=int, default=0, help = '0:DQN, 1:QLearning')
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
    else:
        agent = MultipleReversal().explore().agent
    endpoints = np.zeros((env.height + 2, env.width + 2), dtype = int)
    print(type(agent))
    core(test_gap, train_gap, total_iter)

