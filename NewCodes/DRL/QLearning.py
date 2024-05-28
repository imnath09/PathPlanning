import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(
            self, n_actions=4, 
            learning_rate=0.01, gamma=0.9, 
            e_greedy=0.9):
        self.action_space = range(n_actions)
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.action_space, dtype=np.float64)
        self.random_action = False

    def choose_action(self, observation):
        observation = self.encode_state(observation)
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation,:]
            action = np.random.choice(state_action[state_action==np.max(state_action)].index)
            self.random_action = False
        else:
            action = np.random.choice(self.action_space)
            self.random_action = True
        return action

    def action(self, observation):
        observation = self.encode_state(observation)
        self.check_state_exist(observation)
        state_action = self.q_table.loc[observation,:]
        action = np.random.choice(state_action[state_action==np.max(state_action)].index)
        self.random_action = False
        return action

    def learn(self, s, a, r, s_, done):
        s = self.encode_state(s)
        s_ = self.encode_state(s_)

        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma*self.q_table.loc[s_, :].max()
        self.q_table.loc[s, a] += self.lr*(q_target-q_predict)

    def learn_batch(self, datas):
        for s, a, r, s_, done in datas:
            #print(s, a, r, s_, done)
            self.learn(s, a, r, s_, done)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            #self.q_table = self.q_table.append(pd.Series([0]*len(self.action_space),index=self.q_table.columns,name=state))
            self.q_table.loc[state] = [0.]*len(self.action_space)

    def store_transition(self, *args):
        pass

    def store_transition_batch(self, *args):
        pass

    def encode_state(self, state):
        return '{},{}'.format(state[0], state[1])


