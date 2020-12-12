#visualise rewards and steps from npy

from utils import *


if __name__ == '__main__':	
	# read out npy files
	stats = []
	num_episodes = 2000
	for i in range(3):
		stat = Stats(2000)
		with open('trial' + str(i) + '.npy', 'rb') as f:
			stat.episode_rewards = np.load(f)
			stat.episode_steps = np.load(f)
		stats.append(stat)

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	plot_rewards(ax1, stats, 20)

	ax2 = fig.add_subplot(122)
	plot_steps(ax2, stats, 20)
	plt.savefig('test.png')