import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import copy
#保证神经网络正负记忆都有，最好对半分，才能有一个比较好的结果

# np.random.seed(1)
# torch.manual_seed(1)

# define the network architecture
class Net(nn.Module):
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.el = nn.Linear(n_feature, n_hidden)
		self.q = nn.Linear(n_hidden, n_output)

	def forward(self, x):
		x = self.el(x)
		x = F.relu(x)
		x = self.q(x)
		return x


class DeepQNetwork():#n_actions神经网络输出多少action的值，n_features接收多少observation比如长宽高多少，用feature预测action的值
	def __init__(self, n_actions, n_features, n_hidden=20, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
				replace_target_iter=200, memory_size=500, batch_size=32, e_greedy_increment=None,#output_graph=False
				):
		self.n_actions = n_actions
		self.n_features = n_features
		self.n_hidden = n_hidden
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon_max = e_greedy
		self.replace_target_iter = replace_target_iter#隔多少步后把target参数变成最新的参数
		self.memory_size = memory_size#记忆库容量，记录多少数据
		self.batch_size = batch_size#神经网络提升的时候，会出现随机梯度下降，在神经网络学习的时候会被用到
		self.epsilon_increment = e_greedy_increment#不断缩小随机的范围，好的经历，坏的经历都存储下来(增加探索度或者减少探索度)
		self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

		# total learning step已经学习的步数
		self.learn_step_counter = 0

		# initialize zero memory [s, a, r, s_] 矩阵记忆库，memory_size条记忆，每条记忆长度n_features*2+2
		self.memory = np.zeros((self.memory_size, n_features*2+2))

		self.loss_func = nn.MSELoss()
		# self.loss_func = nn.L1Loss()
		self.cost_his = []#记录下每一步的误差

		self._build_net()#建立神经网络

		self.random_action : bool

		# self.sess = tf.Session()
		#
		# if output_graph:
		# 	# $ tensorboard --logdir=logs
		# 	# tf.train.SummaryWriter soon be deprecated, use following
		# 	tf.summary.FileWriter("logs/", self.sess.graph)
		# self.sess.run(tf.global_variables_initializer())

		# 	def __init__(）函数是一个构造器，用于定义方法中的主要参数。
		# n_actions：用于存储动作集,
		# n_features：用于存储观测值集合,
		# learning_rate：学习率,
		# reward_decay：回报的衰减率,
		# e_greedy：e-greedy方法中的上限值,
		# replace_target_iter：表示更换target-net的步数,
		# memory_size：表示记忆池的大小,
		# batch_size：表示每次更新时从 memory 里面取多少记忆出来,
		# e_greedy_increment=None：epsilon 的增量,
		# output_graph=False,
		#
		# learn_step_counter：记录学习次数 (用于判断是否更换 target_net 参数)
		#
		# 初始化记忆的时候我们初始化为全零，之后我们建立网络，之后将net-eval的值直接复制给net-target。
		

	def _build_net(self):
		self.q_eval = Net(self.n_features, self.n_hidden, self.n_actions)
		self.q_target = Net(self.n_features, self.n_hidden, self.n_actions)
		self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)

		# 两个神经网络是为了固定住一个神经网络 (target_net) 的参数, target_net 是 eval_net 的一个
		# 历史版本, 拥有 eval_net 很久之前的一组参数, 而且这组参数被固定一段时间, 然后再被
		# eval_net 的新参数所替换. 而 eval_net 是不断在被提升的, 所以是一个可以被训练的网络
		# trainable=True. 而 target_net 的 trainable=False.
		#
		# _build_net(self)：首先定义了常量s用于存储当前的s，定义了常量q_target用于存储q_target值，
		# 之后我们使用tensorflow来定义两个网络，之后我们标出损失函数，以及梯度下降的更新方法。
		# 注意的是target-net的参数是由eval-net的参数复制过来的。

	def store_transition(self, s, a, r, s_):
		if not hasattr(self, 'memory_counter'):
			self.memory_counter = 0
		transition = np.hstack((s, [a, r], s_))#在第0行中插入记下transition
		# replace the old memory with new memory
		index = self.memory_counter % self.memory_size#超过数量后，回来覆盖之前的memory，重新存储
		self.memory[index, :] = transition 
		self.memory_counter += 1#进入第2个全0的数组当中
		# store_transition(self, s, a, r, s_)方法用于记录所有的记录，这样就实现了off-policy的目标。
		#
		# 逐条记录，如果记录范围超过了之前设定的范围，我们就进行替换。

	def choose_action(self, observation):
		# 统一 observation 的 shape (1, size_of_observation)
		#observation输入时是一个一维的数据，为了使其能够处理，增加1个维度，变成2维数据
		observation = torch.Tensor(observation[np.newaxis, :])
		if np.random.uniform() < self.epsilon:
			# 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
			#将2维数据放入q_eval神经网络中分析，输出所有行为值
			actions_value = self.q_eval(observation)

			action = np.argmax(actions_value.data.numpy())
			self.random_action = False
		else:
			action = np.random.randint(0, self.n_actions)
			self.random_action = True
		return action
	# 	choose_action(self, observation)方法用于实现e-greedy方法。

	def learn(self):
		# check to replace target parameters
		# 检查是否替换 target_net 参数
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.q_target.load_state_dict(self.q_eval.state_dict())
			#print("\ntarget params replaced\n")

		# sample batch memory from all memory
		# 从 memory 中随机抽取 batch_size 这么多记忆，记忆库中没有这么多，则抽取已经存下来的记忆
		if self.memory_counter > self.memory_size:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
		batch_memory = self.memory[sample_index, :]

		# q_next is used for getting which action would be choosed by target network in state s_(t+1)
		# 获取 q_next (target_net 产生了 q) 和 q_eval(eval_net 产生的 q)

		#运行两个神经网络输出的参数q_next,q_eval，q_target神经网络输出的所有动作值和q_eval神经网络输出的所有值
		#他们的输入是batch_memory中q_next是最后面feature的值，q_eval是最前面feature的值，和怎么存有关，按接收顺序定义
		q_next, q_eval = self.q_target(torch.Tensor(batch_memory[:, -self.n_features:])), self.q_eval(torch.Tensor(batch_memory[:, :self.n_features]))
		# used for calculating y, we need to copy for q_eval because this operation could keep the Q_value that has not been selected unchanged,
		# so when we do q_target - q_eval, these Q_value become zero and wouldn't affect the calculation of the loss 

		# 下面这几步十分重要. q_next, q_eval 包含所有 action 的值,
		# 而我们需要的只是已经选择好的 action 的值, 其他的并不需要.
		# 所以我们将其他的 action 值全变成 0, 将用到的 action 误差值 反向传递回去, 作为更新凭据.
		# 这是我们最终要达到的样子, 比如 q_target - q_eval = [1, 0, 0] - [-1, 0, 0] = [2, 0, 0]
		# q_eval = [-1, 0, 0] 表示这一个记忆中有我选用过 action 0, 而 action 0 带来的 Q(s, a0) = -1, 所以其他的 Q(s, a1) = Q(s, a2) = 0.
		# q_target = [1, 0, 0] 表示这个记忆中的 r+gamma*maxQ(s_) = 1, 而且不管在 s_ 上我们取了哪个 action,
		# 我们都需要对应上 q_eval 中的 action 位置, 所以就将 1 放在了 action 0 的位置.

		# 下面也是为了达到上面说的目的, 不过为了更方面让程序运算, 达到目的的过程有点不同.
		# 是将 q_eval 全部赋值给 q_target, 这时 q_target-q_eval 全为 0,
		# 不过 我们再根据 batch_memory 当中的 action 这个 column 来给 q_target 中的对应的 memory-action 位置来修改赋值.
		# 使新的赋值为 reward + gamma * maxQ(s_), 这样 q_target-q_eval 就可以变成我们所需的样子.
		# 具体在下面还有一个举例说明.


		#计算q现实和q估计比较麻烦，将batch中每一行target最大值取出来，再找eval对应的q值相减，两个argmax不一样
		#针对动作是q估计的动作进行反向传递，而不是针对下一个动作
		q_target = torch.Tensor(q_eval.data.numpy().copy())

		batch_index = np.arange(self.batch_size, dtype=np.int32)
		eval_act_index = batch_memory[:, self.n_features].astype(int)
		reward = torch.Tensor(batch_memory[:, self.n_features+1])
		q_target[batch_index, eval_act_index] = reward + self.gamma*torch.max(q_next, 1)[0]

		"""		#q——target的值算出拉丝，把位置改变，修改矩阵运算，使之能反向传递回去
		        假如在这个 batch 中, 我们有2个提取的记忆, 根据每个记忆可以生产3个 action 的值:
		        q_eval =
		        [[1, 2, 3],
		         [4, 5, 6]]

		        q_target = q_eval =
		        [[1, 2, 3],
		         [4, 5, 6]]

		        然后根据 memory 当中的具体 action 位置来修改 q_target 对应 action 上的值:
		        比如在:
		            记忆 0 的 q_target 计算值是 -1, 而且我用了 action 0;
		            记忆 1 的 q_target 计算值是 -2, 而且我用了 action 2:
		        q_target =
		        [[-1, 2, 3],
		         [4, 5, -2]]
				
		        所以 (q_target - q_eval) 就变成了:
		        [[(-1)-(1), 0, 0],
		         [0, 0, (-2)-(6)]]

		        最后我们将这个 (q_target - q_eval) 当成误差, 反向传递会神经网络.
		        所有为 0 的 action 值是当时没有选择的 action, 之前有选择的 action 才有不为0的值.
		        我们只反向传递之前选择的 action 的值,
		        """

		loss = self.loss_func(q_eval, q_target)
		"""
		optimizer.zero_grad()对应d_weights = [0] * n

		即将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）

		outputs = net(inputs)对应h = dot(input[k], weights)

		即前向传播求出预测的值

		loss = criterion(outputs, labels)对应loss += (label[k] - h) * (label[k] - h) / 2

		这一步很明显，就是求loss（其实我觉得这一步不用也可以，反向传播时用不到loss值，只是为了让我们知道当前的loss是多少）
		loss.backward()对应d_weights = [d_weights[j] + (label[k] - h) * input[k][j] for j in range(n)]

		即反向传播求梯度
		optimizer.step()对应weights = [weights[k] + alpha * d_weights[k] for k in range(n)]

		即更新所有参数
		"""
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# increase epsilon
		self.cost_his.append(loss.detach())# 记录 cost 误差
		# 逐渐增加epsilon，降低行为随机性，由探索到选择最优方案
		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		self.learn_step_counter += 1

		return q_target[batch_index, eval_act_index]

	def plot_cost(self):
		plt.plot(np.arange(len(self.cost_his)), self.cost_his)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.show()

	# Plotting the results for the number of steps
	def plot_results(self, steps, cost):
		# #
		# f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
		# #
		# ax1.plot(np.arange(len(steps)), steps, 'b')
		# ax1.set_xlabel('Episode')
		# ax1.set_ylabel('Steps')
		# ax1.set_title('Episode via steps')
		#
		# #
		# ax2.plot(np.arange(len(cost)), cost, 'r')
		# ax2.set_xlabel('Episode')
		# ax2.set_ylabel('Cost')
		# ax2.set_title('Episode via cost')
		#
		# plt.tight_layout()  # Function to make distance between figures

		#
		plt.figure()
		plt.plot(np.arange(len(steps)), steps, 'b')
		plt.title('Episode via steps')
		plt.xlabel('Episode')
		plt.ylabel('Steps')

		# #
		# plt.figure()
		# plt.plot(np.arange(len(cost)), cost, 'r')
		# plt.title('Episode via cost')
		# plt.xlabel('Episode')
		# plt.ylabel('Cost')

		# Showing the plots
		plt.show()

