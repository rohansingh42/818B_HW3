import matplotlib.pyplot as plt
import gym          # Tested on version gym v. 0.14.0 and python v. 3.17
import glob, os
import tensorflow as tf
import random
import numpy as np
import argparse

# Read user arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train", default=False, help="Flag to Train the network")
args = parser.parse_args()
if args.train == "True":
	training_flag = True
else:
	training_flag = False
print(training_flag)

env = gym.make('MountainCar-v0')
env.seed(42)

PATH = glob.glob(os.path.expanduser('./logs/'))[0]

# Print some info about the environment
print("State space (gym calls it observation space)")
print(env.observation_space)
print("\nAction space")
print(env.action_space)

# Parameters
NUM_STEPS = 200
if training_flag:
	NUM_EPISODES = 5000
else:
	NUM_EPISODES = 1000
LEN_EPISODE = 200
reward_history = []

# Creating perceptron network
n_hidden_1 = 24
n_hidden_2 = 48
n_hidden_3 = 16
n_input = 2			# no. of states
n_output = 3		# no. of actions

# hyperpparams
INIT_LR = 0.5
INIT_EPSILON = 0.4
INIT_GAMMA = 0.99

# Reset graph
tf.reset_default_graph()

# define learning rate
learning_rate = tf.placeholder('float32',
	shape = [], name = 'learning_rate')

# # define weights and biases
# weights = {
# 	'hidden1' : tf.Variable(tf.random_normal
# 		([n_input, n_hidden_1], seed = None, mean = 0, stddev = 1)),
# 	'hidden2' : tf.Variable(tf.random_normal
# 		([n_hidden_1, n_hidden_2], seed = 1, mean = 0, stddev = 1)),
# 	# 'hidden3' : tf.Variable(tf.random_normal
# 	# 	([n_hidden_2, n_hidden_3], seed = 200, mean = 0, stddev = 0.1)),
# 	'output' : tf.Variable(tf.random_normal
# 		([n_hidden_2, n_output], seed = None, mean = 0, stddev = 1)) }

# biases = {
# 	'hidden1' : tf.Variable(tf.random_normal
# 		([n_hidden_1], seed = 1, mean = 0, stddev = 1)),#, trainable = False),
# 	'hidden2' : tf.Variable(tf.random_normal
# 		([n_hidden_2], seed = 1, mean = 0, stddev = 1)),
# 	# 'hidden3' : tf.Variable(tf.random_normal
# 	# 	([n_hidden_3], seed = 200, mean = 0, stddev = 0.1)),
# 	'output' : tf.Variable(tf.random_normal
# 		([n_output], seed = 1, mean = 0, stddev = 1))}#, trainable = False) }

# define layers
input_layer = tf.placeholder(dtype = tf.float32,
	shape = (None, n_input), name = "input_layer")

targetQ = tf.placeholder(dtype = tf.float32,
	shape = (None, n_output), name = "targetQ")

# hidden_layer1 = tf.nn.bias_add(tf.matmul(input_layer,
# 		weights['hidden1']), biases['hidden1'])
# # hidden_layer1 = tf.matmul(input_layer, weights['hidden1'])
# # hidden_layer2 = tf.nn.relu(hidden_layer1)
# hidden_layer1 = tf.nn.leaky_relu(hidden_layer1)

# hidden_layer2 = tf.nn.bias_add(tf.matmul(hidden_layer1,
# 		weights['hidden2']), biases['hidden2'])
# # hidden_layer2 = tf.keras.activations.relu(hidden_layer2)
# hidden_layer2 = tf.nn.leaky_relu(hidden_layer2)

# # hidden_layer3 = tf.add(tf.matmul(hidden_layer2,
# # 		weights['hidden3']), biases['hidden3'])
# # hidden_layer3 = tf.keras.activations.relu(hidden_layer3)

# output_layer = tf.nn.bias_add(tf.matmul(hidden_layer2,
# 		weights['output']), biases['output'])
# # output_layer = tf.matmul(hidden_layer1, weights['output'])
# # output_layer = tf.keras.activations.linear(output_layer)

hidden_layer1 = tf.layers.dense(input_layer, n_hidden_1, activation=tf.nn.leaky_relu, name = 'hidden_layer1')
hidden_layer2 = tf.layers.dense(hidden_layer1, n_hidden_2, activation=tf.nn.leaky_relu, name = 'hidden_layer2')
output_layer = tf.layers.dense(hidden_layer2, n_output, name = 'output_layer')

# define loss function
loss = tf.keras.losses.MSE(targetQ, output_layer)
# loss = tf.compat.v1.losses.huber_loss(targetQ, output_layer)
# loss = tf.reduce_sum(tf.square(targetQ - output_layer))
# loss = tf.keras.losses.binary_crossentropy(targetQ, output_layer)
# loss = tf.nn.softmax_cross_entropy_with_logits(targetQ, output_layer)

# define optimizer algorithm
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
# optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate).minimize(loss)
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

running_av = []

saver = tf.train.Saver()

sess = tf.Session()

writer = tf.summary.FileWriter(PATH, sess.graph)

sess.run(init)

ep_mse = 0
curr_step = 0
n_100 = 0
n_150 = 0
n_175 = 0

if training_flag == False:
	epsilon = 0
	saver.restore(sess, "./model_checkpoints/model_wo_mod2.ckpt")
	# saver.restore(sess, "./model_checkpoints/mcv0_model2.ckpt")

# Run for NUM_EPISODES
for episode in range(NUM_EPISODES):
	
	episode_reward = 0
	ep_mse = 0
	curr_state = env.reset()

	for step in range(LEN_EPISODE):
		# Comment to stop rendering the environment
		# If you don't render, you can speed things up
		if (training_flag == True):
			if ((episode % 100) == 0):
				env.render()
		elif training_flag == False:
			env.render()


		# Randomly sample an action from the action space
		# Should really be your exploration/exploitation policy
		predicted_Q = sess.run(output_layer,
										feed_dict = {
										input_layer : curr_state.reshape(1,n_input),
										learning_rate : lr})

		action = np.argmax(predicted_Q)

		if np.random.rand(1) < epsilon:
			action = np.random.randint(0, 3)
		elif predicted_Q[0,0] == predicted_Q[0,1] and predicted_Q[0,1] == predicted_Q[0,2]:
			action = np.random.randint(0, 3)

		# action = int(action)

		# Step forward and receive next state and reward
		# done flag is set when the episode ends: either goal is reached or
		#       200 steps are done
		next_state, reward, done, _ = env.step(action)

		if training_flag == True:
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

				targetQ_ = predicted_Q.copy()

				# Adjust reward for task completion
				if next_state[0] >= 0.5:
					reward += 10

				targetQ_[0,action] = reward + gamma * np.max(Q_prime)

			o_, loss_, lr_, hl1 = sess.run([optimizer, loss, learning_rate, hidden_layer1],
									feed_dict = {
									input_layer : curr_state.reshape(1,n_input),
									targetQ : targetQ_.reshape(1,n_output),
									learning_rate : lr})
			ep_mse += int(loss_)

		# Record history
		episode_reward += reward

		# Current state for next step
		curr_state = next_state

		
		if done:

			if episode < 100:
				running_av.append(episode_reward)
			else:
				running_av.append(episode_reward)
				del running_av[0]

			if episode_reward > -100:
				n_100 += 1
			if episode_reward > -150:
				n_150 += 1
			if episode_reward > -175:
				n_175 += 1

			rew_100 = sum(running_av)/len(running_av)

			summary = tf.Summary(value = [
					tf.Summary.Value(tag='episode_reward', simple_value=episode_reward),
					tf.Summary.Value(tag='running_reward', simple_value=rew_100),
					tf.Summary.Value(tag='n_100', simple_value=n_100),
					tf.Summary.Value(tag='n_150', simple_value=n_150),
					tf.Summary.Value(tag='n_175', simple_value=n_175),
					tf.Summary.Value(tag='episode_mse', simple_value=ep_mse)])
			
			writer.add_summary(summary, episode)
			
			if curr_state[0] >= 0.5:
				n_successful_eps += 1
				# decay epsilon
				if epsilon >= 0.11:
					epsilon -= 0.04

				# decay learning rate
				if lr >= 1e-3:
					lr *= 0.2

			if training_flag == True:
				print("\nEpisode " + str(episode) + " terminated with - ")
				print("** Reward = " + str(episode_reward))
				print("** Mean Squared Loss = " + str(ep_mse))
				print("** Number of successful episodes = " + str(n_successful_eps))
				print("** Learning Rate = " + str(lr_))
				print("** Epsilon = " + str(epsilon))
			else :
				print("\nEpisode " + str(episode) + " terminated with - ")
				print("** Reward = " + str(episode_reward))
				print("** Running Reward = " + str(rew_100))
				print("** Number of successful episodes = " + str(n_successful_eps))
				if rew_100 > -100:
					print("** Running reward above -100 ")
				if rew_100 > -150:
					print("** Running reward above -150 ")
				if rew_100 > -175:
					print("** Running reward above -175 ")

			break

print("Number of time episode reward was above -100 : ", n_100)
print("Number of time episode reward was above -150 : ", n_150)
print("Number of time episode reward was above -175 : ", n_175)

if training_flag == True:
	save_path = saver.save(sess, "./model_checkpoints/mcv0_model3.ckpt")
