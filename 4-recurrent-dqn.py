import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random 
from collections import namedtuple, deque 
import matplotlib.pyplot as plt 
import gym


from  replay_buffer import ReplayBuffer, ReplayBuffer_LSTM

#Find the device: cpu or gpu
device = torch.device('cuda:0 ' if torch.cuda.is_available() else 'cpu')
print('Device used:', device)

class QNetwork(nn.Module):
	def __init__(self, state_size, action_size, batch_size=64, time_step=8, seed=42, fc1_units=512, fc2_units=512):
		super(QNetwork, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.state_size = state_size
		self.action_size = action_size
		self.lstm = nn.LSTM(input_size=state_size, hidden_size=512,num_layers=1, batch_first=True)
		self.fc1 = nn.Linear(512, fc1_units)
		self.fc2 = nn.Linear(fc1_units, fc2_units)
		self.fc3 = nn.Linear(fc2_units, action_size)
		self.batch_size = batch_size
		self.time_step = time_step


	def forward(self, state, hidden_state, cell_state, batch_size=64, time_step=8):
		if not isinstance(state, torch.Tensor):
			state = torch.from_numpy(state).float()
		#print('state shape:', state.shape)
		state = state.view(batch_size, time_step, self.state_size)
		lstm_out = self.lstm(state, (hidden_state, cell_state))
		
		out = lstm_out[0][:, time_step-1, :]
		h_n = lstm_out[1][0]
		c_n = lstm_out[1][1]

		x = F.relu(self.fc1(out))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x, (h_n, c_n)


	def init_hidden_states(self, batch_size=64):
		h = torch.zeros(1, batch_size, 512).float().to(device)
		c = torch.zeros(1, batch_size, 512).float().to(device)
		
		return h, c

class RDQN_Agent:
	def __init__(self, state_size, action_size, batch_size=64, time_step = 8, seed=42, learning_rate=1e-4):
		self.state_size = state_size
		self.action_size = action_size
		self.seed = random.seed(seed)
		self.batch_size = batch_size
		self.time_step = time_step
		self.qnetwork_local = QNetwork(state_size, action_size, seed=seed).to(device)
		self.qnetwork_target = QNetwork(state_size, action_size, seed=seed).to(device)
		self.soft_update(self.qnetwork_local, self.qnetwork_target, 1.0)
		self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

		self.memory = ReplayBuffer_LSTM(action_size, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, time_step=time_step, seed=seed)
		self.t_step = 0
		self.learning_count = 0

	def step(self, state, action, reward, next_state, done):
		self.memory.add(state, action, reward, next_state, done)
		self.t_step = (self.t_step + 1) % UPDATE_EVERY
		if self.t_step == 0:
			if len(self.memory) > BATCH_SIZE:
				experiences = self.memory.sample()
				self.learn(experiences, GAMMA)


	def step_lstm(self, episode):
		#print('Memory len: ', len(self.memory))
		self.t_step = (self.t_step + 1) % UPDATE_EVERY
		if len(episode) == 0:
			if self.t_step == 0:
				if len(self.memory) > BATCH_SIZE:
					batch = self.memory.get_batch(self.batch_size, self.time_step)
					self.learn_lstm(batch, GAMMA)
		else:
			self.memory.add_episode(episode)
			if self.t_step == 0:
				if len(self.memory) > BATCH_SIZE:
					batch = self.memory.get_batch(self.batch_size, self.time_step)
					self.learn_lstm(batch, GAMMA)


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

	
	def act_lstm(self, state, hidden_state, cell_state, batch_size=64, time_step=8, eps=0.):
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		#set network to eval mode
		self.qnetwork_local.eval()
		with torch.no_grad():
			#print(batch_size, time_step)
			action_values, lstm_hiddens = self.qnetwork_local(state, hidden_state, cell_state, batch_size, time_step)
		#set network back to train mode
		self.qnetwork_local.train()
		hidden_state = lstm_hiddens[0]
		cell_state = lstm_hiddens[1]
		if random.random() > eps:
			return np.argmax(action_values.cpu().data.numpy()), hidden_state, cell_state
		else:
			return random.choice(np.arange(self.action_size)), hidden_state, cell_state


	def learn(self, experiences, gamma):
		states, actions, rewards, next_states, dones = experiences

		Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

		Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
		Q_expected = self.qnetwork_local(states).gather(1, actions)

		loss = F.mse_loss(Q_expected, Q_targets)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
	
	
	def learn_lstm(self, batch, gamma):
		#print('Learning')
		self.learning_count += 1
		hidden_batch, cell_batch = self.qnetwork_local.init_hidden_states()
		#states, actions, rewards, next_states, dones = experiences
		states = []
		actions = []
		rewards = []
		next_states = []
		dones = []
		for episode in batch:
			s, a, r, n_s, d = [], [], [], [], []
			for tup in episode:
				s.append(tup[0])
				a.append(tup[1])
				r.append(tup[2])
				n_s.append(tup[3])
				d.append(tup[4])
			states.append(s)	
			actions.append(a)
			rewards.append(r)
			next_states.append(n_s)
			dones.append(d)
		
		states = np.array(states)
		actions = np.array(actions)
		rewards = np.array(rewards)
		next_states = np.array(next_states)
		dones = np.array(dones)
	
		torch_states = torch.from_numpy(states).float().to(device)
		torch_actions = torch.from_numpy(actions).long().to(device)
		torch_rewards = torch.from_numpy(rewards).float().to(device)
		torch_next_states = torch.from_numpy(next_states).float().to(device)
		torch_dones = torch.from_numpy(dones).float().to(device)
		
		Q_targets_next, _ = self.qnetwork_target(torch_next_states, hidden_batch, cell_batch)#.detach().max(1)[0].unsqueeze(1)
		Q_targets_next = Q_targets_next.detach().max(1)[0].unsqueeze(1)
	
		#print(torch_rewards[:, TIME_STEP-1].unsqueeze(dim=1).shape)
		#print(Q_targets_next.shape)
		#print(torch_dones[:, TIME_STEP-1].unsqueeze(dim=1).shape)
		
		Q_targets = torch_rewards[:,TIME_STEP-1].unsqueeze(dim=1) + gamma * Q_targets_next * (1 - torch_dones[:, TIME_STEP-1].unsqueeze(dim=1))
		Q_expected, _  = self.qnetwork_local(torch_states, hidden_batch, cell_batch)#.gather(1, actions)
		#print(torch_actions[:, TIME_STEP-1].unsqueeze(dim=1).shape)
		Q_expected = Q_expected.gather(1, torch_actions[:, TIME_STEP-1].unsqueeze(dim=1))
	
		assert Q_expected.shape == Q_targets.shape, 'Mismatched dimension'
		loss = F.mse_loss(Q_expected, Q_targets)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)



	def soft_update(self, local_model, target_model, tau):
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


	def dqn_train(self, n_episodes=20000, max_t = 500, eps_start=1.0, eps_end=0.001, eps_decay=0.999):
		scores = []
		steps = []
		scores_window = deque(maxlen=20)
		eps = eps_start

		for i_episode in range(1, n_episodes+1):
			#print('\rEpisode {}',i_episode)
			state = env.reset()
			score = 0
			step = 0
			count = 0
			local_memory = []
			hidden_state, cell_state = self.qnetwork_local.init_hidden_states(batch_size=1)
			for t in range(max_t):
				#print('\r Episode {} Step {}'.format(i_episode, t))
				action, hidden_state, cell_state = self.act_lstm(state, hidden_state, cell_state, batch_size=1, time_step=1, eps=eps)
				next_state, reward, done, info = env.step(action)
				local_memory.append((state, action, reward, next_state, done))
				#self.step(state, action, reward, next_state, done)
				self.step_lstm([])
				state = next_state 
				score += reward 
				step = t
				if done:
					#self.step_lstm(local_memory)	
					break
			#self.step_lstm(local_memory)	
			self.memory.add_episode(local_memory)
			scores_window.append(score)
			scores.append(score)
			steps.append(step)
			eps = max(eps_end, eps_decay * eps)
			if i_episode % 500 == 0:
				print('\rEpisode {} Eps {} \tAverage Score {:.2f}'.format(i_episode, eps, np.mean(scores_window)))
				print('Everage steps: ', np.mean(steps[-100:]))
		torch.save(self.qnetwork_local.state_dict(), './results/recurrent_dqn_cart_pole.pth')
		print('The agent trained {} times'.format(self.learning_count))
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
	BUFFER_SIZE = int(20000)
	BATCH_SIZE = 64 
	TIME_STEP = 8
	GAMMA = 0.99
	TAU = 0.001
	LR = 0.00025 
	UPDATE_EVERY = 5 


	env = gym.make('CartPole-v0')
	#env = gym.make('Taxi-v2')
	state_size = len(env.reset())
	#state_size = env.observation_space.n
	num_actions = env.action_space.n

	print('State size: ', state_size)
	print('Action size: ', num_actions)

	q_network = QNetwork(state_size, num_actions)
	print(q_network)
	#state0 = env.reset()
	#print('First qnetwork output: ', q_network(state0))


	dqn_agent = RDQN_Agent(state_size=state_size, action_size=num_actions,learning_rate=LR)
	scores, steps = dqn_agent.dqn_train()
	print('Done training')
	env.close()
	eps = list(range(len(scores)))

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax1.plot(eps, smooth_curve(scores, 20))
	ax1.set_ylabel('Score')
	ax1.set_xlabel('Episode #')
	
	ax2 = fig.add_subplot(122)
	ax2.plot(eps, smooth_curve(scores, 20))
	ax2.set_ylabel('Steps')
	ax2.set_xlabel('Episode #')

	plt.savefig('./results/recurrent_dqn_cart_pole.png')
