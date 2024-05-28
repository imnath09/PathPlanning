
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

#给 UAVs 用

class ExperienceReplayDQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.01 #0.001
        self.model = self._build_model()
        self.memory = deque(maxlen = 2000)

        self.batch_size = 32
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.01  # 最小探索率

        self.epsilon_decay = 0.999  # 探索率衰减率

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # 返回具有最大Q值的动作

    def learn(self):
        if len(self.memory) <= self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 定义环境
class Environment:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def step(self, state, action):
        # 执行动作并返回下一个状态和奖励
        next_state = ...
        reward = ...
        done = ...
        return next_state, reward, done

# 定义状态空间和动作空间的大小
state_size = ...
action_size = ...

# 创建环境和Agent
env = Environment(state_size, action_size)
agent = ExperienceReplayDQN(state_size, action_size)

# 训练Agent
for episode in range(1000):
    state = ...
    total_reward = 0
    for time_step in range(100):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(state, action)

        agent.store_transition(state, action, reward, next_state, done)

        state = next_state

        agent.learn()
        total_reward += reward
        if done:
            break

