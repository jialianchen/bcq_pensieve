import argparse
import copy
import importlib
import json
import os
import h5py
import numpy as np
import torch
from test_bcq_env import Env
import bcq as discrete_BCQ

name_list = ['sb','ab','nsb','rb','db']

S_DIM = [6, 8]
A_DIM = 6
test_file_num = 142
# Trains BCQ offline
def train_BCQ(replay_buffer, num_actions, state_dim, device, args, parameters):
	# For saving files
	setting = f"{args.env}_{args.seed}"
	buffer_name = f"{args.buffer_name}_{setting}"

	# Initialize and load policy
	policy = discrete_BCQ.discrete_BCQ(
	
		num_actions,
		state_dim,
		device,
		args.BCQ_threshold,
		parameters["discount"],
		parameters["optimizer"],
		parameters["optimizer_parameters"],
		parameters["polyak_target_update"],
		parameters["target_update_freq"],
		parameters["tau"],
		parameters["initial_eps"],
		parameters["end_eps"],
		parameters["eps_decay_period"],
		parameters["eval_eps"]
	)

	# Load replay buffer	
	#replay_buffer.load(f"./buffers/{buffer_name}")
	
	evaluations = []
	episode_num = 0
	done = True 
	training_iters = 0
	#print(replay_buffer[0])
	while training_iters < args.max_timesteps: 
		
		for _ in range(int(parameters["eval_freq"])):
			policy.train(replay_buffer)

		evaluations.append(eval_policy(policy, args.env, args.seed))
		np.save(f"./results/BCQ_{setting}", evaluations)

		training_iters += int(parameters["eval_freq"])
		print(f"Training iterations: {training_iters}")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = Env()
	

	avg_reward = 0.
	reward_list = []
	for _ in range(test_file_num):
		state, done = eval_env.reset(), False
		r_batch = []
		while not done:
			action = policy.select_action(np.array(state), eval=True)
			state, reward, done, _ = eval_env.step(action)
		
			r_batch.append(reward)
			if done:
				reward_list.append(np.mean(r_batch))
	avg_reward = np.mean(reward_list)

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":



	regular_parameters = {
		# Exploration
		"start_timesteps": 1e3,
		"initial_eps": 0.1,
		"end_eps": 0.1,
		"eps_decay_period": 1,
		# Evaluation
		"eval_freq": 5e3, #5e3
		"eval_eps": 0,
		# Learning
		"discount": 0.99,
		"buffer_size": 1e6,
		"batch_size": 64,
		"optimizer": "Adam",
		"optimizer_parameters": {
			"lr": 3e-4
		},
		"train_freq": 1,
		"polyak_target_update": True,
		"target_update_freq": 1,
		"tau": 0.005
	}

	# Load parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="PongNoFrameskip-v0")     # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_name", default="Default")        # Prepends name to filename
	parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment or train for
	parser.add_argument("--BCQ_threshold", default=0.3, type=float)# Threshold hyper-parameter for BCQ
	parser.add_argument("--low_noise_p", default=0.2, type=float)  # Probability of a low noise episode when generating buffer
	parser.add_argument("--rand_action_p", default=0.2, type=float)# Probability of taking a random action when generating buffer, during non-low noise episode
	parser.add_argument("--train_behavioral", action="store_true") # If true, train behavioral policy
	parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
	args = parser.parse_args()
	
	print("---------------------------------------")	
	if args.train_behavioral:
		print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
	elif args.generate_buffer:
		print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
	else:
		print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if args.train_behavioral and args.generate_buffer:
		print("Train_behavioral and generate_buffer cannot both be true.")
		exit()

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists("./models"):
		os.makedirs("./models")

	if not os.path.exists("./buffers"):
		os.makedirs("./buffers")

	# Make env and determine properties

	parameters =  regular_parameters

	# # Set seeds
	# env.seed(args.seed)
	# env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Initialize buffer
	#replay_buffer =  #utils.ReplayBuffer(state_dim, is_atari, atari_preprocessing, parameters["batch_size"], parameters["buffer_size"], device)

	_file = name_list[0]+'.h5'
	f = h5py.File(_file, 'r')
	#print(len(list(f.keys())))
	length = len(list(f.keys()))
	f.close()
	replay_buffer = []
	for i in range(length):
		tmp_list = []
		for idx in range(len(name_list)):
			s_name = name_list[idx]
			h5py_file = s_name+'.h5'
			
			f = h5py.File(h5py_file, 'r')
			tmp_list.append(np.array((f[str(i)])))
		replay_buffer.append(tmp_list)

	# if args.train_behavioral or args.generate_buffer:
	# 	interact_with_environment(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters)
	# else:
	train_BCQ(replay_buffer, A_DIM,S_DIM, device, args, parameters)