import matplotlib.pyplot as plt
import gym          # Tested on version gym v. 0.14.0 and python v. 3.17
import glob, os
import tensorflow as tf
import random
import numpy as np
# from tensorboard import SummaryWriter
# from gymbag import record_hdf5, HDF5Reader

env = gym.make('MountainCar-v0')
# env.seed(42)

# PATH = glob.glob(os.path.expanduser('./logs/'))[0]
# writer = tf.summary.SummaryWriter('{}{}'.format(PATH, datetime.now().strftime('%b%d_%H-%M-%S')))

# Print some info about the environment
print("State space (gym calls it observation space)")
print(env.observation_space)
print("\nAction space")
print(env.action_space)

# Parameters
NUM_STEPS = 200
NUM_EPISODES = 10000
LEN_EPISODE = 200
reward_history = []

## Creating perceptron network
n_hidden_1 = 64
n_hidden_2 = 64
n_hidden_3 = 16
n_input = 2			# no. of states
n_output = 3		# no. of actions

# hyperpparams

INIT_LR = 1
INIT_EPSILON = 0.4
INIT_GAMMA = 0.95


# Reset graph
tf.reset_default_graph()

# global lrstep
# lrstep = 0
# define learning rate
learning_rate = tf.placeholder('float32',
	shape = [], name = 'learning_rate')
# learning_rate = tf.train.exponential_decay(0.0002, lrstep, 200, 0.99, staircase = True)
# lrr = tf.placeholder('float32', shape = [], name = 'lrr')

# # define weights and biases
# weights = {
# 	'hidden1' : tf.Variable(tf.random_normal
# 		([n_input, n_hidden_1], seed = None, mean = 0, stddev = 1)),
# 	# 'hidden2' : tf.Variable(tf.random_normal
# 	# 	([n_hidden_1, n_hidden_2], seed = 1, mean = 0, stddev = 1)),
# 	# 'hidden3' : tf.Variable(tf.random_normal
# 	# 	([n_hidden_2, n_hidden_3], seed = 200, mean = 0, stddev = 0.1)),
# 	'output' : tf.Variable(tf.random_normal
# 		([n_hidden_1, n_output], seed = None, mean = 0, stddev = 1)) }

# biases = {
# 	'hidden1' : tf.Variable(tf.random_normal
# 		([n_hidden_1], seed = 1, mean = 0, stddev = 1), trainable = False),
# 	# 'hidden2' : tf.Variable(tf.random_normal
# 	# 	([n_hidden_2], seed = 1, mean = 0, stddev = 1), trainable = False),
# 	# 'hidden3' : tf.Variable(tf.random_normal
# 	# 	([n_hidden_3], seed = 200, mean = 0, stddev = 0.1)),
# 	'output' : tf.Variable(tf.random_normal
# 		([n_output], seed = 1, mean = 0, stddev = 1), trainable = False) }

# define layers
input_layer = tf.placeholder(dtype = tf.float32,
	shape = (None, n_input), name = "input_layer")

targetQ = tf.placeholder(dtype = tf.float32,
	shape = (None, n_output), name = "targetQ")

# # hidden_layer1 = tf.nn.bias_add(tf.matmul(input_layer,
# # 		weights['hidden1']), biases['hidden1'])
# # hidden_layer1 = tf.nn.bias_add(tf.matmul(input_layer,
# 		# weights['hidden1']), biases['hidden1'])
# hidden_layer1 = tf.matmul(input_layer, weights['hidden1'])
# hidden_layer2 = tf.nn.relu(hidden_layer1)

# # hidden_layer2 = tf.nn.bias_add(tf.matmul(hidden_layer1,
# # 		weights['hidden2']), biases['hidden2'])
# # hidden_layer2 = tf.keras.activations.relu(hidden_layer2)

# # hidden_layer3 = tf.add(tf.matmul(hidden_layer2,
# # 		weights['hidden3']), biases['hidden3'])
# # hidden_layer3 = tf.keras.activations.relu(hidden_layer3)

# # output_layer = tf.nn.bias_add(tf.matmul(hidden_layer1,
# # 		weights['output']), biases['output'])
# output_layer = tf.matmul(hidden_layer2, weights['output'])
# # output_layer = tf.keras.activations.linear(output_layer)

hidden_layer1 = tf.layers.dense(input_layer, n_hidden_1, activation=tf.nn.leaky_relu, name = 'hidden_layer1')
# hidden_layer2 = tf.layers.dense(hidden_layer1, n_hidden_2, activation=tf.nn.relu, name = 'hidden_layer2')
output_layer = tf.layers.dense(hidden_layer1, n_output, name = 'output_layer')

# defining a tensor variable giving theindex of the
# max of all obtained Q values, giving the action
# maxQ = tf.argmax(output_layer, 1)

# define loss function
# loss = tf.keras.losses.MSE(targetQ, output_layer)
loss = tf.compat.v1.losses.huber_loss(targetQ, output_layer)
# loss = tf.reduce_sum(tf.square(targetQ - output_layer))
# loss = tf.keras.losses.binary_crossentropy(targetQ, output_layer)
# loss = tf.nn.softmax_cross_entropy_with_logits(targetQ, output_layer)

# define optimizer algorithm
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

# optimizer = tf.train.AdamOptimizer().minimize(loss)
# optimizer = tf.train.AdagradOptimizer(learning_rate = 0.1).minimize(loss)

# variable initializer
init = tf.global_variables_initializer()

lr = INIT_LR
epsilon = INIT_EPSILON
gamma = INIT_GAMMA
n_successful_eps = 0
curr_state = env.reset()
sc = 0

meansqerr = []

running_av = []

n_175 = []
n_150 = []
n_100 = []

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)

	ep_mse = []
	# Run for NUM_EPISODES
	for episode in range(NUM_EPISODES):
		episode_reward = 0
		
		curr_state = env.reset()
		# print("\nStarting state : ", curr_state)
		max_position = -0.4
		for step in range(LEN_EPISODE):
			# # Comment to stop rendering the environment
			# # If you don't render, you can speed things up
			if episode % 100 == 0:
				env.render()

			# Randomly sample an action from the action space
			# Should really be your exploration/exploitation policy
			predicted_Q = sess.run(output_layer,
											feed_dict = {
											input_layer : curr_state.reshape(1,n_input),
											learning_rate : lr})
											# lrr : episode})

			action = np.argmax(predicted_Q)
			# print(predicted_Q)
			if np.random.rand(1) < epsilon:
				action = np.random.randint(0, 10)
				if action < 5:
					action = 0
				elif action > 5:
					action = 2
				else:
					action = 1
				# print(action)
			elif predicted_Q[0,0] == predicted_Q[0,1] and predicted_Q[0,1] == predicted_Q[0,2]:
				action = np.random.randint(0, 3)

			# action = int(action)

			# Step forward and receive next state and reward
			# done flag is set when the episode ends: either goal is reached or
			#       200 steps are done
			next_state, reward, done, _ = env.step(action)

			# This is where your NN/GP code should go
			# Create target vector
			# Train the network/GP
			# Update the policy
			if next_state is None:
				targetQ_ = predicted_Q
			else:
				Q_prime = sess.run(output_layer,
										feed_dict = {
										input_layer : next_state.reshape(1,n_input),
										learning_rate : lr})
										# lrr : episode})

				targetQ_ = predicted_Q.copy()
				# print(Q_prime)
				# targetQ_[0, action[0]] = (1-lr)*targetQ_[0, action[0]] + lr*(reward + gamma * np.max(Q_prime))
				
				# reward = curr_state[0] + 0.5
		
				# # Keep track of max position
				# if curr_state[0] > max_position:
				# 	max_position = curr_state[0]
				
				# # Adjust reward for task completion
				# if curr_state[0] >= 0.5:
				# 	reward += 2

				# if next_state[0] > -0.4 and next_state[0] < 0:
				# 	reward += 2
				# elif next_state[0] >= 0 and next_state[0] < 0.3:
				# 	reward += 6
				# elif next_state[0] >= 0.3:
				# 	reward += 11
				# print(targetQ_)
				# print(reward, predicted_Q, np.max(Q_prime))
				targetQ_[0,action] = reward + gamma * np.max(Q_prime)
				# print(targetQ_ - predicted_Q)
				# print(action, targetQ_ - predicted_Q, targetQ_, predicted_Q)

			# o_, loss_, w, lr_, hl1 = sess.run([optimizer, loss, weights, learning_rate, hidden_layer1],
			# 						feed_dict = {
			# 						input_layer : curr_state.reshape(1,n_input),
			# 						targetQ : targetQ_.reshape(1,n_output),
			# 						learning_rate : lr})			
			# 						# lrr : episode})
			o_, loss_, lr_, hl1 = sess.run([optimizer, loss, learning_rate, hidden_layer1],
									feed_dict = {
									input_layer : curr_state.reshape(1,n_input),
									targetQ : targetQ_.reshape(1,n_output),
									learning_rate : lr})			
									# lrr : episode})
			

			# Record history
			episode_reward += reward

			# Current state for next step
			curr_state = next_state
			# print(curr_state)

			ep_mse.append(loss_)
			
			if done:

				# print(lr_)

				# print(predicted_Q)
				# print(w)
				# print(np.max(hl1))

				# update epsilon
				# epsilon *= 0.99**(int(episode/100))

				# Record history
				# reward_history.append(episode_reward)

				# You may want to plot periodically instead of after every episode
				# Otherwise, things will slow down
				# if (episode % 500) == 0:
				# 	fig = plt.figure(1)
				# 	plt.clf()
				# 	plt.xlim([0,NUM_EPISODES])
				# 	plt.plot(reward_history,'ro')
				# 	plt.plot([rv/len(running_av) for rv in running_av],'bo')
				# 	plt.xlabel('Episode')
				# 	plt.ylabel('Reward')
				# 	plt.title('Reward Per Episode')
				# 	plt.pause(0.01)
				# 	fig.canvas.draw()

				# 	fig = plt.figure(2)
				# 	plt.clf()
				# 	plt.xlim([0,NUM_EPISODES])
				# 	plt.plot([emse/len(ep_mse) for emse in ep_mse],'bo')
				# 	plt.xlabel('Episode')
				# 	plt.ylabel('MSE')
				# 	plt.title('MSE Per Episode')
				# 	plt.pause(0.01)
				# 	fig.canvas.draw()

				if episode % 500 == 0 and n_successful_eps == 0:
					epsilon *= 0.5
				if curr_state[0] >= 0.5:# and sc < 10:
					n_successful_eps += 1
					if epsilon > 0.1:
						epsilon *= 0.5
					if lr > 1e-4:
						lr *= 0.2
					# sc += 1

				# if episode < 100:
				# 	running_av.append(episode_reward)
				# else:
				# 	running_av.append(episode_reward)
				# 	del running_av[0]

				print("\nEpisode " + str(episode) + " terminated with - ")
				print("** Reward = " + str(episode_reward))
				print("** Mean Squared Loss = " + str(loss_))
				print("** Number of successful episodes = " + str(n_successful_eps))
				print("** Learning Rate = " + str(lr_))
				print("** Epsilon = " + str(epsilon))

				# tf.summary.scalar('ep loss', np.sum(ep_mse), episode)
				# tf.summary.scalar('ep reward', episode_reward, episode)
				# tf.summary.scalar('weights', W, episode)
				# tf.summary.scalar('biases', b, episode)
				# tf.summary.scalar('learning_rate', lr_, episode)
				# tf.summary.scalar('epsilon', epsilon, episode)



				break

	save_path = saver.save(sess, "model_wo_mod2.ckpt")
