
import os
import sys
sys.path.append('..')
import argparse

from DRL.QLearning import *
from DRL.VallinaDQN import *
from DRL.ReplayDQNTorch import *
from DRL.PriorDQN import *

from Algs.SPaRM import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--testgap', type=int, default=10)
    parser.add_argument('--traingap', type=int, default=200)
    parser.add_argument('--iter', type=int, default=300)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--model', type=str, default='ql')
    # QLearningTable() # VallinaDQN() # ReplayDQN() # PriorDQN()
    parser.add_argument('--alg', type=str, default='rfe')
    # UAVs # RFE # SpacePartition # SPaRM #
    args = parser.parse_args()

    seed = args.seed
    test_gap = args.testgap
    train_gap = args.traingap
    total_iter = args.iter
    horizon = args.horizon
    model = args.model
    alg = args.alg
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    # 设置 Python 内置的 random 模块的随机种子
    random.seed(seed)
    # 设置torch随机种子
    torch.manual_seed(seed)
    # 如果使用 GPU，还需要设置 CUDA 的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU

    # QLearningTable() # VallinaDQN() # ReplayDQN() # PriorDQN()
    if model in 'qlearning':
        agent = QLearningTable()
    elif model in 'vallina':
        agent = VallinaDQN()
    elif model in 'replay':
        agent = ReplayDQN()
    elif model in 'prior':
        agent = PriorDQN()

    env = GymMaze()#map=np.zeros((10, 10))) # GymMaze(map=np.zeros((10, 10))) # 

    # UAVs # RFE # SpacePartition # SPaRM #
    if alg in 'uavs':
        procedure = UAVs(agent, env)
    elif alg in 'rfe':
        procedure = RFE(agent, env)
    elif alg in 'spacepartition':
        procedure = SpacePartition(agent, env)
    elif alg in 'sparm':
        procedure = SPaRM(agent, env)


    procedure.explore(horizon)
    procedure.Planning(test_gap, train_gap, total_iter, horizon)


'''
常用随机数
0
1
42
1234
2020
2021

# 设置 NumPy 的随机种子
np.random.seed(seed)

# 设置 Python 内置的 random 模块的随机种子
random.seed(seed)

# 设置随机种子
seed = 42
torch.manual_seed(seed)

# 如果使用 GPU，还需要设置 CUDA 的随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU

在某些情况下，为了确保完全的可重复性，你还可以禁用 CuDNN 的确定性算法：
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
'''
