import gym
import argparse

from DQN_Agent import *

import sys
sys.path.append('..')

from Maze.gymmaze import *

# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 10000 # Episode limitation 10000
STEP = 300 # Step limitation in an episode 300
TEST = 10 # The number of experiment test every 100 episode 10

def main():
    for episode in range(1, 1 + EPISODE):
        # initialize task
        state = env.reset()
        # Train
        for _ in range(STEP):
        #done = False
        #while not done:
            action = agent.egreedy_action(state) # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            # Define reward for agent
            reward_agent = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for _ in range(TEST):
                state = env.reset()
                for _ in range(STEP):
                #done = False
                #while not done:
                    #env.render()
                    action = agent.action(state) # direct action for test
                    state, reward, done, _ = env.step(action)
                    #print('{},{},{},{}'.format(state,action,reward,done))
                    #print(type(state), type(action), type(reward))
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
            #if ave_reward >= 300: # 200:
            #    break
    agent.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action = 'store_true', help = 'render or not')
    parser.add_argument(
        '--mode', type = int, choices = [0, 1],
        help = '0:cartpole-v0, 1:maze')
    args = parser.parse_args()
    mode = args.mode
    rendered = args.render

    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME) if mode == 0 else UnrenderedMaze() if not rendered else GymMaze()

    if mode == 0:
        agent = DQN(env.observation_space.shape[0], env.action_space.n)
    else:
        agent = DQN(env.observation_space_n, env.action_space_n)

    main()

