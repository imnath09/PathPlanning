import torch
import torch.nn as nn
import torch.nn.functional as F
#import wandb

import random
import numpy as np

class SumTree:
    def __init__(self, capacity: int):
        # 初始化SumTree，设定容量
        self.capacity = capacity
        # 数据指针，指示下一个要存储数据的位置
        self.data_pointer = 0
        # 数据条目数
        self.n_entries = 0
        # 构建SumTree数组，长度为(2 * capacity - 1)，用于存储树结构
        self.tree = np.zeros(2 * capacity - 1)
        # 数据数组，用于存储实际数据
        self.data = np.zeros(capacity, dtype=object)

    def update(self, tree_idx, p):#更新采样权重
        # 计算权重变化
        change = p - self.tree[tree_idx]
        # 更新树中对应索引的权重
        self.tree[tree_idx] = p

        # 从更新的节点开始向上更新，直到根节点
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, p, data):#向SumTree中添加新数据
        # 计算数据存储在树中的索引
        tree_idx = self.data_pointer + self.capacity - 1
        # 存储数据到数据数组中
        self.data[self.data_pointer] = data
        # 更新对应索引的树节点权重
        self.update(tree_idx, p)

        # 移动数据指针，循环使用存储空间
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        # 维护数据条目数
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, v):#采样数据
        # 从根节点开始向下搜索，直到找到叶子节点
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            # 如果左子节点超出范围，则当前节点为叶子节点
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                # 根据采样值确定向左还是向右子节点移动
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        # 计算叶子节点在数据数组中的索引
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total(self):
        return int(self.tree[0])

class PERContainer:#ReplayTree for the per(Prioritized Experience Replay) DQN. 
    def __init__(self, capacity):
        self.capacity = capacity # 记忆回放的容量
        self.tree = SumTree(capacity)  # 创建一个SumTree实例
        self.abs_err_upper = 1.  # 绝对误差上限
        self.epsilon = 0.01
        ## 用于计算重要性采样权重的超参数
        self.beta_increment_per_sampling = 0.001
        self.alpha = 0.6
        self.beta = 0.4 
        self.abs_err_upper = 1.

    def __len__(self):# 返回存储的样本数量
        return self.tree.total()

    def push(self, error, sample):#Push the sample into the replay according to the importance sampling weight
        p = (np.abs(error.detach().numpy()) + self.epsilon) ** self.alpha
        self.tree.add(p, sample)         


    def sample(self, batch_size):
        pri_segment = self.tree.total() / batch_size

        priorities = []
        batch = []
        idxs = []

        is_weights = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total() 

        for i in range(batch_size):
            a = pri_segment * i
            b = pri_segment * (i+1)

            s = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
            prob = p / self.tree.total()

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return zip(*batch), idxs, is_weights
    
    def batch_update(self, tree_idx, abs_errors):#Update the importance sampling weight
        abs_errors += self.epsilon

        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class PriorDQN(object):
    '''Prioritized Experience Replay'''
    def __init__(
            self, n_state=2, n_action=4,
            learning_rate = 0.001, gamma = 0.9,
            replace_target_iter = 100, 
            epsilon_start = 0.5, epsilon_min = 0.01,
            buffer_capacity = 100000, batch_size = 100, 
            ):
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.n_state = n_state
        self.n_action = n_action
        self.eval_net = DQN(self.n_state, self.n_action).to(self.device)
        self.target_net = DQN(self.n_state, self.n_action).to(self.device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr = learning_rate)
        self.loss_func = nn.MSELoss()

        self.action_space = range(n_action)
        self.memory = PERContainer(capacity = buffer_capacity)
        self.batch_size = batch_size

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_min
        self.replace_target_iter = replace_target_iter
        self.learn_step_counter = 0
        self.memory_counter = 0      

    def choose_action(self, state):
        if np.random.uniform() >= self.epsilon:
            action = np.random.choice(self.action_space)
        else:#随机
            state = torch.unsqueeze(torch.FloatTensor(state).to(self.device), 0)# to device 是否在这
            actions_value=self.eval_net.forward(state)
            action=torch.max(actions_value,1)[1].data.numpy()
            action=action[0]

        return action
    
    def action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state).to(self.device), 0)# to device 是否在这
        actions_value=self.eval_net.forward(state)
        action=torch.max(actions_value,1)[1].data.numpy()
        action=action[0]
        return action

    def learn(self, *args):
        if self.memory_counter < self.batch_size:
            return 
        #if self.learn_step_counter % self.replace_target_iter==0:
        #    self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        batch, tree_idx,is_weights = self.memory.sample(self.batch_size)
        b_s, b_a, b_r, b_s_ = batch
        b_s, b_a, b_r, b_s_ = np.array(b_s), np.array(b_a), np.array(b_r), np.array(b_s_)
        #ssss=np.array(b_s)
        #print(ssss, type(ssss))
        b_s = torch.FloatTensor(b_s).to(self.device)
        b_a = torch.unsqueeze(torch.LongTensor(b_a).to(self.device), 1)
        b_r = torch.unsqueeze(torch.FloatTensor(b_r).to(self.device), 1)
        b_s_ = torch.FloatTensor(b_s_).to(self.device)
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

        loss = (torch.FloatTensor(is_weights).to(self.device) * self.loss_func(q_eval, q_target)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        abs_errors = torch.abs(q_eval - q_target).detach().numpy().squeeze()
        self.memory.batch_update(tree_idx, abs_errors)  # 更新经验的优先级
        # 更新 epsilon
        #if self.epsilon > self.epsilon_end:
        #    self.epsilon = self.epsilon * 0.999

    def learn_batch(self, *args):
        self.learn()

    def store_transition(self, state, action, reward, next_state, done):
        policy_val =self.eval_net(torch.FloatTensor(state).to(self.device))[action]
        target_val =self.target_net(torch.FloatTensor(next_state).to(self.device))
        transition = (state, action, reward, next_state)

        if done:
            error = abs(policy_val-reward)
        else:
            error = abs(policy_val - reward - self.gamma * torch.max(target_val))
        self.memory.push(error, transition)  # 添加经验和初始优先级
        self.memory_counter += 1

    def store_transition_batch(self, datas):
        for state, action, reward, next_state, done in datas:
            self.store_transition(state, action, reward, next_state, done)


# 测试代码
if __name__ == '__main__':
    import gym
    env=gym.make("CartPole-v1",render_mode="human") 

    dqn=PriorDQN(env.observation_space.shape[0], env.action_space.n)
    for i in range(100):
        print('<<<<<<<Episode:%s'%i)
        s=env.reset()
        episode_reward_sum=0

        while True:
            a=dqn.choose_action(s)
            s_,r,done,info=env.step(a)
            x,x_dot,theta,theta_dot=s_
            r1=(env.x_threshold-abs(x))/env.x_threshold-0.8
            r2=(env.theta_threshold_radians-abs(theta))/env.theta_threshold_radians-0.5
            new_r=r1+r2
            dqn.store_transition(s,a,new_r,s_,done)
            episode_reward_sum+=new_r
            s=s_
            if done:
                print('episode%s---reward_sum:%s'%(i,round(episode_reward_sum,2)))
                break

        dqn.learn()
