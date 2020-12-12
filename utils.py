#Implement RandomWalk class and SubgoalDiscovery class
import random
import copy
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


def argmax_rand(arr):
	# np.argmax with random tie breaking
	return np.random.choice(np.flatnonzero(np.isclose(arr, np.max(arr), atol=1e-3)))

def detect_state_change(state1, state2):
	for i in range(len(state1)):
		if state1[i] != state2[i]:
			return True

	return False

def one_hot_encoding(val, total_num):
	vector = [0 for i in range(total_num)]
	vector[val - 1] = 1

	return vector


def state_preprocessing(state):
	new_state = list(state) 
	for host in range(0, 15+1):
		new_state[7 + 5*host] = one_hot_encoding(state[7 + 5*host], 3)
		new_state[10 + 5*host] = one_hot_encoding(state[10 + 5*host], 5)

	new_state = flatten(new_state)
	return new_state


def flatten(state):
	flatten = []
	for i in range(len(state)):
		if not isinstance(state[i], list):
			flatten.append(state[i])
		else:
			for j in range(len(state[i])):
				flatten.append(state[i][j])

	return flatten


class RandomWalk():
	def __init__(self,
				env=None,
				max_steps=2000,
				subgoal_discovery=None,
				experience_memory=None,
				**kwargs):
		self.env = env
		self.max_steps = max_steps 
		self.max_episodes = 21 
		self.subgoal_discovery = subgoal_discovery
		self.experience_memory = experience_memory
		self.subgoal_dicovery_freq = 25
		self.__dict__.update(kwargs)

	def walk(self):
		print('#'*60)
		print('Random walk for Subgoal Discovery')
		print('#'*60)
		for i in range(self.max_episodes+1):
			print('Episode: ', i)
			s = state_preprocessing(self.env.reset_red())
			
			for j in range(self.max_steps):			
				a = random.choice(self.env.get_red_actions())
				sp, r, terminal, step_info = self.env.perform_red_action(a)
				#(state, action, reward, next_state, done)
				sp = state_preprocessing(sp)
				e = (s,a,r,sp,terminal)
				info = step_info.split()	
				
				if ('Bruteforce' in info) and ('succeeded!' in info):
					r += 0.01
				
				
				self.experience_memory.push(e)
				if r>0:
					print('r: {}, info: {}'.format(r, step_info))
					print('Pushing outlier')
					self.subgoal_discovery.push_outlier(tuple(sp))
				
				if terminal:
					print('Flag captured at: ', j)
					break
				s = copy.copy(sp)

			if i>0 and i%10 == 0:
				self.subgoal_discovery.find_kmeans_clusters()
				# self.subgoal_discovery.report()

	def walk_and_find_doorways(self):
		print('#'*60)
		print('Random walk for Doorways Type Subgoal Discovery')
		print('#'*60)
		for i in range(self.max_episodes):
			s = self.env.reset()
			for j in range(self.max_steps):			
				a = self.env.action_space.sample()
				sp, r, terminal, step_info = self.env.step(a)
				e = (s,a,r,sp)
				self.subgoal_discovery.push_doorways(e)
				if terminal:
					break
				s = copy.copy(sp)



class SubgoalDiscovery():
	def __init__(self,
				n_clusters=4,
				experience_memory=None,
				kmeans=None,
				**kwargs):
		self.n_clusters = n_clusters
		self.experience_memory = experience_memory
		self.outliers = set() 
		self.centroid_memory= [] # all the centroids over time of learning
		self.centroid_subgoals = [] # list of recent centroids
		self.G = [] # recent list of all subgoals
		self.C = None # Kmeans centroids in numpy arrays rounded
		self.X = None
		self.kmeans = kmeans
		self.doorways = []
		self.doorway_pairs = []

		self.__dict__.update(kwargs)
		
	def feed_data(self,X):
		self.X = X

	def find_kmeans_clusters(self):
		if self.X is None and self.experience_memory is None:
			print('Error! No data to work with, either feed_data or pass memory')

		if self.experience_memory is not None:
			self.X = self.experience_memory.X

		if self.C is None:
			# print('first time of using kmeans to find centroids')
			init = 'random' 
		else:
			# print('updating Kmeans centroids using previous centroids')
			init = self.C
		if self.kmeans is None:
			self.kmeans = KMeans(n_clusters=self.n_clusters,init=init,max_iter=300)
		self.kmeans.fit(self.X)
		self.C = self.cluster_centroids()
		self.centroid_memory.append(self.C)		
		self.centroid_subgoals = [ tuple(g) for g in list(self.C) ]
		self.G = self.centroid_subgoals + list(self.outliers)

	def find_kmeans_clusters_random_seed(self):
		self.X = self.experience_memory.X
		self.kmeans = KMeans(n_clusters=self.n_clusters,init='random',max_iter=300)
		self.kmeans.fit(self.X)
		self.C = self.cluster_centroids()
		self.centroid_memory.append(self.C)		
		self.centroid_subgoals = [ tuple(g) for g in list(self.C) ]
		self.G = self.centroid_subgoals + list(self.outliers)

	def find_gaussian_clusters(self):
		self.gaussian = GaussianMixture(n_components=4).fit(self.X)

	def find_kmeans_clusters_online(self,init='k-means++'):
		self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters,init=init,max_iter=300).fit(self.X)

	def cluster_centroids(self):
		#return np.round_(self.kmeans.cluster_centers_).astype(int)
		return np.round_(self.kmeans.cluster_centers_).astype(int)

	def predict_closest_cluster(self, s):
		c_id = self.predict_closest_cluster_index(s)
		centroids = self.kmeans.cluster_centers_
		c = centroids[:,c_id]
		return np.round_(c*16+0.5)

	def predict_closest_cluster_index(self, s):
		z = np.array(list(s)).reshape(1,-1)
		return self.kmeans.predict(z)

	def push_outlier(self, outlier,threshold=1):
		if len(self.outliers) == 0:
			self.outliers.add(outlier)
			self.G = self.centroid_subgoals + list(self.outliers)
			print('self.outliers updated')	
			return 	

		distance = []
		for member in self.outliers:
			temp_diff = abs((np.array(member) - np.array(outlier)).sum())
			distance.append(temp_diff)
		
		if min(distance) > 3:
			self.outliers.add(outlier)
			self.G = self.centroid_subgoals + list(self.outliers)
			
		print('self.outliers updated')	
		# 	print('outlier discovered: ', outlier)
		# else:
		# 	print('discovered outlier already in the outliers list')

	def push_doorways(self,e,threshold=5):
		s = e[0]
		a = e[1]
		r = e[2]
		sp =e[3]
		room_1 = self.predict_closest_cluster_index(s)
		room_2 = self.predict_closest_cluster_index(sp)
		if room_1 != room_2:
			if len(self.doorways)==0:
				self.doorways.append(s)
				self.doorways.append(sp)
				self.doorway_pairs.append([s,sp])
				print('doorways discovered: ', s, 'and',sp)
			else:
				distance = []
				for member in self.doorways:
					distance.append( (member[0]-s[0])**2+(member[1]-s[1])**2 )
				if min(distance) >= threshold:
					self.doorways.append(s)
					self.doorways.append(sp)
					print('doorways discovered: ', s, 'and',sp)
					self.doorway_pairs.append([s,sp])
				# else:
				# 	print('discovered doorways already in the doorways list')

	def report(self):
		print('outliers: ', self.outliers) 
		print('centroids: ', self.centroid_subgoals)
		print('doorways: ', self.doorways)


class Stats():
    def __init__(self, num_episodes=20000, num_states = 6, continuous=False):
        self.episode_rewards = np.zeros(num_episodes)
        self.episode_lengths = np.zeros(num_episodes)
        if not continuous:
            self.visitation_count = np.zeros((num_states, num_episodes))
            self.target_count = np.zeros((num_states, num_episodes))

def plot_rewards(ax, episodes_ydata, smoothing_window = 100, c='b'):
    #smoothing_window = 100

    overall_stats_q_learning = []
    for trialdata in episodes_ydata:
        overall_stats_q_learning.append(pd.Series(trialdata.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean())
        #overall_stats_q_learning.append(pd.Series(trialdata.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean().data)
    m_stats_q_learning = np.mean(overall_stats_q_learning, axis=0)
    std_stats_q_learning = np.std(overall_stats_q_learning, axis=0)

    ax.plot(range(len(m_stats_q_learning)), m_stats_q_learning, c=c)
    ax.fill_between(range(len(std_stats_q_learning)), m_stats_q_learning - std_stats_q_learning, m_stats_q_learning + std_stats_q_learning, alpha=0.5, edgecolor=c, facecolor=c)


def plot_steps(ax, episodes_ydata, smoothing_window = 100, c='g'):
    #smoothing_window = 100

    overall_stats_q_learning = []
    for trialdata in episodes_ydata:
        overall_stats_q_learning.append(pd.Series(trialdata.episode_lengths).rolling(smoothing_window, min_periods=smoothing_window).mean())
        #overall_stats_q_learning.append(pd.Series(trialdata.episode_lengths).rolling(smoothing_window, min_periods=smoothing_window).mean().data)
    m_stats_q_learning = np.mean(overall_stats_q_learning, axis=0)
    std_stats_q_learning = np.std(overall_stats_q_learning, axis=0)

    ax.plot(range(len(m_stats_q_learning)), m_stats_q_learning, c=c)
    ax.fill_between(range(len(std_stats_q_learning)), m_stats_q_learning - std_stats_q_learning, m_stats_q_learning + std_stats_q_learning, alpha=0.5, edgecolor=c, facecolor=c)


def plot_visitation_counts(episodes_ydata, smoothing_window = 1000, c=['b', 'g', 'r', 'y', 'k', 'c'], num_states = None):

    if not num_states: 
        num_states = len(episodes_ydata[0].visitation_count)

    overall_stats_q_learning = [[] for i in range(num_states)]
    for trialdata in episodes_ydata:
        for state in range(num_states):
            overall_stats_q_learning[state].append(pd.Series(trialdata.visitation_count[state]).rolling(smoothing_window, min_periods=smoothing_window).mean().data)
    
    for state in range(num_states):
        m_stats_q_learning = np.mean(overall_stats_q_learning[state], axis=0)
        std_stats_q_learning = np.std(overall_stats_q_learning[state], axis=0)

        plt.plot(range(len(m_stats_q_learning)), m_stats_q_learning, c=c[state])
        plt.fill_between(range(len(std_stats_q_learning)), m_stats_q_learning - std_stats_q_learning, m_stats_q_learning + std_stats_q_learning, alpha=0.5, edgecolor=c[state], facecolor=c[state])

def plot_target_counts(episodes_ydata, smoothing_window = 1000, c=['b', 'g', 'r', 'y', 'k', 'c']):

    num_states = len(episodes_ydata[0].target_count)

    overall_stats_q_learning = [[] for i in range(num_states)]
    for trialdata in episodes_ydata:
        for state in range(num_states):
            overall_stats_q_learning[state].append(pd.Series(trialdata.target_count[state]).rolling(smoothing_window, min_periods=smoothing_window).mean().data)
    
    for state in range(num_states):
        m_stats_q_learning = np.mean(overall_stats_q_learning[state], axis=0)
        std_stats_q_learning = np.std(overall_stats_q_learning[state], axis=0)

        plt.plot(range(len(m_stats_q_learning)), m_stats_q_learning, c=c[state])
        plt.fill_between(range(len(std_stats_q_learning)), m_stats_q_learning - std_stats_q_learning, m_stats_q_learning + std_stats_q_learning, alpha=0.5, edgecolor=c[state], facecolor=c[state])

def plot_q_values(model, observation_space, action_space):

    res = 100

    test_observations = np.linspace(observation_space.low, observation_space.high, res)
    
    print((action_space.n, res))
    q_values = np.zeros((action_space.n, res))

    for action in range(action_space.n):
        for obs in range(res):
            q_values[action, obs] = model.predict(test_observations[obs])[0, action]

        plt.plot(test_observations, q_values[action])
