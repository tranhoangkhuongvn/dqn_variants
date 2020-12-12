#Credit to mynkpl1998
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random 
from collections import namedtuple, deque 
import matplotlib.pyplot as plt 

from parameters import *

class ReplayBuffer:
	"""
	Fixed size buffer to store experience tuples
	"""
	def __init__(self, action_size, buffer_size, batch_size, device, seed):
		"""
		Params:
			action_size (int): dimension of each action
			buffer_size (int): maximum size of buffer
			batch_size (int): size of each training batch
			seed (int): random seed
		"""
		#self.action_size = action_size
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state","done"])
		self.seed = random.seed(seed)
		self.device = device

	def add(self, state, action, reward, next_state, done):
		"""Add a new experience to memory"""
		e = self.experience(state, action, reward, next_state, done)
		self.memory.append(e)


	def sample(self):
		"""Randomly sample a batch of experiences from memory"""
		experiences = random.sample(self.memory, k =self.batch_size)
		
		
		states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
		return (states, actions, rewards, next_states, dones)

	def __len__(self):
		"""Return the current size of internal memory"""
		return len(self.memory)




class ReplayBuffer_LSTM:
	def __init__(self,  buffer_size, sequence_length=1, batch_size=32, device='cpu', seed=42):
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.experience = deque(maxlen=buffer_size)
		self.seed = random.seed(seed)
		self.sequence_length = sequence_length 
		self.device = device
		self.episode_buffer = []
		self.mdp_tuple = namedtuple("MDP_Tuple", field_names=["state", "action", "reward", "next_state","done"])
	
	
	def add(self, state, action, reward, next_state, done):
		self.episode_buffer.append(self.mdp_tuple(state, action, reward, next_state, done))	
		if done == True:
			self.experience.append(np.vstack(self.episode_buffer))
			self.episode_buffer = []
	
	def sample(self):
		episodes = random.sample(self.experience, k=self.batch_size)
		# episodes is list of list:  batch_size x episode_len 	
		batch_states = []
		batch_actions = []
		batch_rewards = []
		batch_next_states = []
		batch_dones = []
		for episode in episodes:
			ep_len = len(episode)
			end = max(ep_len - self.sequence_length, 1)
			seq_start = np.random.randint(0, end)
			trajectory = episode[seq_start : seq_start + self.sequence_length]	

			states = torch.from_numpy(np.stack([t[0] for t in trajectory if t is not None])).float().to(self.device)
			actions = torch.from_numpy(np.stack([t[1] for t in trajectory if t is not None])).long().to(self.device)
			rewards = torch.from_numpy(np.stack([t[2] for t in trajectory if t is not None])).float().to(self.device)
			next_states = torch.from_numpy(np.stack([t[3] for t in trajectory if t is not None])).float().to(self.device)
			dones = torch.from_numpy(np.stack([t[4] for t in trajectory if t is not None])).float().to(self.device)
			batch_states.append(states)
			batch_actions.append(actions)
			batch_rewards.append(rewards)
			batch_next_states.append(next_states)
			batch_dones.append(dones)

		batch_states = np.array(batch_states)	
		batch_actions = np.array(batch_actions)	
		batch_rewards = np.array(batch_rewards)	
		batch_next_states = np.array(batch_next_states)	
		batch_dones = np.array(batch_dones)	


		return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

	
	def __len__(self):
		return len(self.experience)



