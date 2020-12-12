import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random 
from collections import namedtuple, deque 
import matplotlib.pyplot as plt 
import gym


from  replay_buffer import ReplayBuffer

#Find the device: cpu or gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device used:', device)

class Dueling_QNetwork(nn.Module):
	def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=512):
		super(Dueling_QNetwork, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.feature = nn.Sequential(
			nn.Linear(state_size, fc1_units),
			nn.ReLU()
		)

		self.advantage = nn.Sequential(
			nn.Linear(fc1_units, fc2_units),
			nn.ReLU(),
			nn.Linear(fc2_units, action_size),
		)

		self.value = nn.Sequential(
			nn.Linear(fc1_units, fc2_units),
			nn.ReLU(),
			nn.Linear(fc2_units, 1),
		)


	def forward(self, state):
		x = self.feature(state)
		advantage = self.advantage(x)
		value = self.value(x)
		# if value.shape == torch.Size([1,1]):
		#	print('Value shape {0} at this state is {1} '.format(value.shape, value))
		result = value + advantage - advantage.mean()
		return result


class DDQN_Agent:
	def __init__(self, state_size, action_size, seed, learning_rate=1e-4):
		self.state_size = state_size
		self.action_size = action_size
		self.seed = random.seed(seed)

		self.qnetwork_local = Dueling_QNetwork(state_size, action_size, seed).to(device)
		self.qnetwork_target = Dueling_QNetwork(state_size, action_size, seed).to(device)
		self.soft_update(self.qnetwork_local, self.qnetwork_target, 1.0)
		self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

		self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
		self.t_step = 0


	def step(self, state, action, reward, next_state, done):
		self.memory.add(state, action, reward, next_state, done)
		self.t_step = (self.t_step + 1) % UPDATE_EVERY
		if self.t_step == 0:
			if len(self.memory) > BATCH_SIZE:
				experiences = self.memory.sample()
				self.learn(experiences, GAMMA)

	def act(self, state, eps=0.):
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		#set network to eval mode
		self.qnetwork_local.eval()
		with torch.no_grad():
			action_values = self.qnetwork_local(state)
		#set network back to train mode
		self.qnetwork_local.train()

		if random.random() > eps:
			return np.argmax(action_values.cpu().data.numpy())
		else:
			return random.choice(np.arange(self.action_size))

	
	def learn(self, experiences, gamma):
		states, actions, rewards, next_states, dones = experiences

		#Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

		q_temp = self.qnetwork_local(next_states).detach()
		_, a_max = q_temp.max(1)	
		
		Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, a_max.unsqueeze(1))
		#print(Q_targets_next.shape, dones.shape)
		Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
		Q_expected = self.qnetwork_local(states).gather(1, actions)
		#print(Q_expected.shape, Q_targets.shape)
		assert Q_expected.shape == Q_targets.shape, "Mismatched shape"
		loss = F.mse_loss(Q_expected, Q_targets)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


	def soft_update(self, local_model, target_model, tau):
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


	def dqn_train(self, n_episodes=2000, max_t = 500, eps_start=1.0, eps_end=0.001, eps_decay=0.995):
		scores = []
		steps = []
		scores_window = deque(maxlen=20)
		eps = eps_start

		for i_episode in range(1, n_episodes+1):
			state = env.reset()
			score = 0
			step = 0
			count = 0
			for t in range(max_t):
				action = self.act(state, eps)
				next_state, reward, done, info = env.step(action)

				self.step(state, action, reward, next_state, done)
				state = next_state 
				score += reward 
				step = t
				if done:
					break
			scores_window.append(score)
			scores.append(score)
			steps.append(step)
			eps = max(eps_end, eps_decay * eps)
			if i_episode % 500 == 0:
				print('\rEpisode {} \tAverage Score {:.2f}'.format(i_episode, np.mean(scores_window)))
				print('Everage steps: ', np.mean(steps[-100:]))
		torch.save(self.qnetwork_local.state_dict(), './results/dueling_ddqn_cart_pole.pth')

		return scores, steps


def smooth_curve(inputs, I):
	episodes = len(inputs)
	avg_inputs = [inputs[0]]
	for i in range(1, episodes):
		interval = min(i, I)
		avg_input = np.average(inputs[i - interval: i])
		avg_inputs.append(avg_input)

	return avg_inputs



if __name__ == '__main__':
	BUFFER_SIZE = int(100000)
	BATCH_SIZE = 64
	GAMMA = 0.99
	TAU = 0.001
	LR = 1e-4
	UPDATE_EVERY = 5


	env = gym.make('CartPole-v0')
	state_size = len(env.reset())
	num_actions = env.action_space.n

	print('State size: ', state_size)
	print('Action size: ', num_actions)

	#q_network = QNetwork(state_size, num_actions, 42)
	#print(q_network)
	#state0 = env.reset()
	#print('First qnetwork output: ', q_network(state0))


	dqn_agent = DDQN_Agent(state_size=state_size, action_size=num_actions, seed=42, learning_rate=LR)
	scores, steps = dqn_agent.dqn_train()
	print('Done training')
	env.close()
	eps = list(range(len(scores)))

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax1.plot(eps, smooth_curve(scores, 20))
	ax1.set_ylabel('Score')
	ax1.set_xlabel('Episode #')
	ax1.grid()	

	ax2 = fig.add_subplot(122)
	ax2.plot(eps, smooth_curve(scores, 20))
	ax2.set_ylabel('Steps')
	ax2.set_xlabel('Episode #')
	ax2.grid()
	plt.savefig('./results/dueling_ddqn_cart_pole.png')
