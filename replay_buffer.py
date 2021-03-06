#Credit to mynkpl1998
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random 
from collections import namedtuple, deque 
import matplotlib.pyplot as plt 
import gym

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
	def __init__(self, action_size, buffer_size, batch_size, device='cpu', seed=0):
		self.action_size = action_size
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.experience = namedtuple('Experience', 
									field_names=['state', 'action', 'reward', 'next_state', 'done'])
		self.seed = random.seed(seed)
		self.device = device

	
	def add(self, state, action, reward, next_state, done):
		e = self.experience(state, action, reward, next_state, done)
		self.memory.append(e)


	def sample(self):
		experiences = random.sample(self.memory, k=self.batch_size)
		states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

		return (states, actions, rewards, next_states, dones)

	
	def __len__(self):
		return len(self.memory)


class ReplayBuffer_LSTM:
	def __init__(self, action_size, buffer_size, batch_size=64, time_step = 8, seed=42):
		self.action_size = action_size
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.experience = namedtuple('Experience', 
									field_names=['state', 'action', 'reward', 'next_state', 'done'])
		self.seed = random.seed(seed)

	
	def add_episode(self, episode):
		self.memory.append(episode)

	
	def get_batch(self, batch_size, time_step):
		sampled_episodes = random.sample(self.memory, batch_size)	
		batch = []
		for episode in sampled_episodes:
			point = np.random.randint(0, len(episode) + 1 - time_step)
			batch.append(episode[point: point+time_step])

		return batch
	
	def __len__(self):
		return len(self.memory)



