
import sys
sys.path.append('..')

from Maze.gymmaze import *
from Algorithm.Sarsa import SarsaLambdaTable

import matplotlib.pyplot as plt
import argparse

TOTAL_EPISODE = 50000
STAT_GAP = 1000

x = []
#e1 = []
#e2 = []
c = []
def SarsaTest():
    suc_matrix = np.array([[0, 0],[0, 0]])
    pl = 0
    for episode in range(1, TOTAL_EPISODE + 1):
        observation = env.reset()
        explore = False # 探索

        action = RL.choose_action(str(observation))
        RL.eligibility_trace *= 0

        while True:
            observation_, reward, done, info = env.step(action)
            env.render()

            action_ = RL.choose_action(str(observation_))
            if RL.random_action:
                explore = True

            RL.learn(str(observation), action, reward, str(observation_), action_, done)
            observation = observation_
            action = action_

            if done:
                i = 0 if explore else 1 # 0是探索，1是利用
                suc_matrix[i][0] += 1

                if reward == 1: #
                    pl += env.cur_path.__len__()
                    suc_matrix[i][1] += 1
                    if info is not None: # 新的最佳解决方案
                        print("episode{} {}".format(episode, info))
                break

        if episode % STAT_GAP == 0:
            print("episode:{}; explore:{}/{}={}; exploit:{}/{}={}".format(
                episode,
                suc_matrix[0][1], suc_matrix[0][0], round(suc_matrix[0][1] / suc_matrix[0][0], 4),
                suc_matrix[1][1], suc_matrix[1][0], round(suc_matrix[1][1] / suc_matrix[1][0], 4)))
            x.append(episode)
            #e1.append(suc_matrix[0][1] / suc_matrix[0][0])
            #e2.append(suc_matrix[1][1] / suc_matrix[1][0])
            c.append(pl / (suc_matrix[0][1] + suc_matrix[1][1]))
            suc_matrix *= 0
            pl = 0

    print('game over')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action = 'store_true', help = 'render or not')
    args = parser.parse_args()
    rendered = args.render
    env = GymMaze() if rendered else UnrenderedMaze()
    RL = SarsaLambdaTable(actions=env.action_space)
    SarsaTest()

    RL.show_q_table()
    print(env.endpoints)
    plt.plot(x, c, 'g-')
    #plt.plot(x, e1, 'g-')
    #plt.plot(x, e2, 'b-')
    plt.grid()
    plt.show()


