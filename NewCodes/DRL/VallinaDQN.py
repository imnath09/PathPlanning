import torch
import torch.nn as nn

import numpy as np
#给UAV Swarm用

class Net(nn.Module):
    def __init__(self, state_size, action_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VallinaDQN:
    def __init__(
            self, n_state=2, n_action=4, 
            learning_rate=0.001, gamma=0.9, 
            epsilon_start=0.5, epsilon_min=0.01, epsilon_decay=0.999
            ):
        self.state_size = n_state
        self.action_size = n_action
        self.action_space = range(n_action)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = Net(n_state, n_action)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        if np.random.uniform() >= self.epsilon:
            return np.random.choice(self.action_space)# random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.model(state_tensor)
                return torch.argmax(q_values).item()

    def action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        target_f = self.model(state_tensor)
        #print(target_f)
        target_f[0][action] = target

        loss = self.loss_fn(self.model(state_tensor), target_f.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #if self.epsilon > self.epsilon_min:
        #    self.epsilon *= self.epsilon_decay

    def learn_batch(self, datas):
        '''学习一组数据，每个参数都是列表而不是单个数据。'''
        for state, action, reward, next_state, done in datas:
            #print(state, action, reward, next_state, done)
            self.learn(state, action, reward, next_state, done)

    def store_transition(self, *args):
        pass

    def store_transition_batch(self, *args):
        pass

# 示例应用
if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = VallinaDQN(state_size, action_size)

    EPISODES = 1000
    for e in range(EPISODES):
        state = env.reset()
        for time in range(500):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
                break
