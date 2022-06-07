from dabeijing import Environment
from RL_brain import DeepQNetwork

import numpy as np
import time
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
#from Algorithm.TorchDQN import DeepQNetwork

def run_maze():
	step = 0
	steps = []
	all_costs = []
	start1 = time.perf_counter()
	path = [0]
	c = 0
	for episode in range(20000):
		#print("episode: {}".format(episode))
		observation = dabeijing.reset()
		i = 0
		cost = 0
		rd = False
		while True:
			#print("step: {}".format(step))
			dabeijing.render()
			action = RL.choose_action(observation)
			observation_, reward, done, info = dabeijing.step(action)
			RL.store_transition(observation, action, reward, observation_)
			if (step>200) and (step%5==0):
				RL.learn()
			observation = observation_

			rd |= RL.random_action
			i += 1
			if done:
				if not rd:
					c += 1
					if info == 'goal':
						lenth = len(dabeijing.d)
						path.append(lenth)
				steps += [i]
				all_costs += [cost]
				break
			step += 1
	end1 = time.perf_counter()
	print('game over')
	print("运行耗时", end1 - start1)
	dabeijing.final()
	# dabeijing.destroy()
	# RL.print_q_table()
	plt.figure()
	plt.title('{} success/{} exploit'.format(len(path) - 1, c))
	plt.plot(path, 'b.')
	plt.show()
	
	plt.figure()
	plt.title('steps')
	plt.plot(np.arange(len(steps)), steps, 'b')
	plt.show()



if __name__ == '__main__':
	# start1 = time.perf_counter()
	dabeijing = Environment()
	RL = DeepQNetwork(dabeijing.n_actions, dabeijing.n_features,
					learning_rate=0.01,
					reward_decay=0.9,
					e_greedy=0.9,
					replace_target_iter=200,
					memory_size=2000
					# memory_size = 500
					#output_graph=True
					)
	dabeijing.after(100, run_maze)
	dabeijing.mainloop()
	# end1 = time.perf_counter()
	# print("运行耗时", end1 - start1)
	RL.plot_cost()


	"""
	首先是导入相应的库，本例子中是导入了环境maze_env和强化学习更新方法RL_brain。之后定义一个方法run_maze()用于更新。
	方法的内容如下：先是定义了总共eposide的循环次数300，之后随机的初始化一个状态值s，再通过e-greedy的方法去选择一个动作a，
	之后获得即时奖励，并且进入到下一个状态，将（st，at，r，st+1）放入到记忆池中，之后再在池中抽样一部分用于学习，并且更新网络的参数，
	其中前200步是不学习的，同时每隔五步去学习一次。学习之后，我们将下一步状态转化为当前状态。直到q表变得收敛，我们就停止循环。

	在main函数中，我们显示定义了强化学习的运行环境，之后再定义了强化学习算法DQN的参数，之后我们开始循环，并且计算参数了。
	"""
	#每五步学习一次，可以降低关联性