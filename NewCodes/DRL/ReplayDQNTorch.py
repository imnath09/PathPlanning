import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

# 定义Q网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# 定义DQN代理
class ReplayDQN:
    def __init__(
            self, n_state=2, n_action=4,
            #learning_rate=0.001, gamma=0.99, 
            learning_rate=0.001, gamma=0.9, 
            epsilon_start=0.9, epsilon_min=0.01, epsilon_decay=0.999,
            buffer_capacity=100000, batch_size=100, 
            ):
        self.state_dim = n_state
        self.action_dim = n_action
        self.q_network = QNetwork(n_state, n_action)
        self.target_q_network = QNetwork(n_state, n_action)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.action_space = range(n_action)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size

        self.gamma = gamma
        self.epsilon = epsilon_start# 1.0
        self.epsilon_min = epsilon_min


        #self.update_target_network()

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        #print(state, q_values, q_values.argmax().item())
        return q_values.argmax().item()

    def choose_action(self, state):
        if np.random.uniform() >= self.epsilon:
            action = np.random.choice(self.action_space)
            return action
        else:# < epsilon
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()

    def learn(self, *args):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_network(states)
        next_q_values = self.target_q_network(next_states)
        target_q_values = q_values.clone()

        for i in range(self.batch_size):
            target_q_values[i, actions[i]] = rewards[i] + self.gamma * next_q_values[i].max() * (1 - dones[i])

        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #if self.epsilon > self.epsilon_min:
        #    self.epsilon *= 0.999

    def learn_batch(self, *args):
        self.learn()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def store_transition_batch(self, datas):
        for state, action, reward, next_state, done in datas:
            self.store_transition(state, action, reward, next_state, done)

class YourPathPlanningEnv:
    pass


# 示例：路径规划
def main():
    env = YourPathPlanningEnv()  # 这里需要实现您的路径规划环境
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ReplayDQN(state_dim, action_dim)

    num_episodes = 500
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.learn()

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        agent.update_target_network()

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
