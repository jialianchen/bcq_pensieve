from cmath import pi
import copy
from matplotlib.pyplot import pink


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
A_DIM = 6
def convert_torch(variable, dtype=np.float32):
    if variable.dtype != dtype:
        variable = variable.astype(dtype)

    return torch.from_numpy(variable)
# # Used for Atari
# class Conv_Q(nn.Module):
# 	def __init__(self, frames, num_actions):
# 		super(Conv_Q, self).__init__()
# 		self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
# 		self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
# 		self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

# 		self.q1 = nn.Linear(3136, 512)
# 		self.q2 = nn.Linear(512, num_actions)

# 		self.i1 = nn.Linear(3136, 512)
# 		self.i2 = nn.Linear(512, num_actions)


# 	def forward(self, state):
# 		c = F.relu(self.c1(state))
# 		c = F.relu(self.c2(c))
# 		c = F.relu(self.c3(c))

# 		q = F.relu(self.q1(c.reshape(-1, 3136)))
# 		i = F.relu(self.i1(c.reshape(-1, 3136)))
# 		i = self.i2(i)
# 		return self.q2(q), F.log_softmax(i, dim=1), i


# # Used for Box2D / Toy problems
# class FC_Q(nn.Module):
# 	def __init__(self, state_dim, num_actions):
# 		super(FC_Q, self).__init__()
# 		self.q1 = nn.Linear(state_dim, 256)
# 		self.q2 = nn.Linear(256, 256)
# 		self.q3 = nn.Linear(256, num_actions)

# 		self.i1 = nn.Linear(state_dim, 256)
# 		self.i2 = nn.Linear(256, 256)
# 		self.i3 = nn.Linear(256, num_actions)		


# 	def forward(self, state):
# 		q = F.relu(self.q1(state))
# 		q = F.relu(self.q2(q))

# 		i = F.relu(self.i1(state))
# 		i = F.relu(self.i2(i))
# 		i = self.i3(i)
# 		return self.q3(q), F.log_softmax(i, dim=1), i
class Pen_Q(nn.Module):
	def __init__(self, s_dim, a_dim):
		super(Pen_Q, self).__init__()
		self.s_dim = s_dim
		self.a_dim = a_dim

		self.fc1 = nn.Linear(1, 128)
		self.fc2 = nn.Linear(1, 128)
		self.conv1 = nn.Conv1d(1, 128, kernel_size=4)
		self.conv2 = nn.Conv1d(1, 128, kernel_size=4)
		self.conv3 = nn.Conv1d(1, 128, kernel_size=4)
		self.fc3 = nn.Linear(1, 128)
		self.out_linear = nn.Linear(2048, A_DIM)


		self.ifc1 = nn.Linear(1, 128)
		self.ifc2 = nn.Linear(1, 128)
		self.iconv1 = nn.Conv1d(1, 128, kernel_size=4)
		self.iconv2 = nn.Conv1d(1, 128, kernel_size=4)
		self.iconv3 = nn.Conv1d(1, 128, kernel_size=4)
		self.ifc3 = nn.Linear(1, 128)
		self.iout_linear = nn.Linear(2048, A_DIM)


	def forward(self, inputs):
		  # print(inputs[:, 4:5, :A_DIM])
        # print(inputs[:, 4:5, :A_DIM].size())
		split_0 = F.relu(self.fc1(inputs[:, 0:1, -1]))
		split_1 = F.relu(self.fc2(inputs[:, 1:2, -1]))
		split_2 = F.relu(self.conv1(inputs[:, 2:3, :].view(-1, 1, self.s_dim[1])))
    	# split_2 = F.relu(self.conv1(inputs[:, 2:3, :]))
		split_3 = F.relu(self.conv2(inputs[:, 3:4, :].view(-1, 1, self.s_dim[1])))
		split_4 = F.relu(self.conv3(inputs[:, 4:5, :A_DIM].view(-1, 1, A_DIM)))
		split_5 = F.relu(self.fc3(inputs[:, 5:6, -1]))
		split_2_flatten, split_3_flatten, split_4_flatten = split_2.flatten(start_dim=1), split_3.flatten(start_dim=1), split_4.flatten(start_dim=1)
        # print(split_2_flatten.size(), split_3_flatten.size())
		merge_net = torch.cat([split_0, split_1, split_2_flatten, split_3_flatten, split_4_flatten, split_5], dim=1)
		logits = self.out_linear(merge_net)


		isplit_0 = F.relu(self.ifc1(inputs[:, 0:1, -1]))
		isplit_1 = F.relu(self.ifc2(inputs[:, 1:2, -1]))
		isplit_2 = F.relu(self.iconv1(inputs[:, 2:3, :].view(-1, 1, self.s_dim[1])))
		isplit_3 = F.relu(self.iconv2(inputs[:, 3:4, :].view(-1, 1, self.s_dim[1])))
		isplit_4 = F.relu(self.iconv3(inputs[:, 4:5, :A_DIM].view(-1, 1, A_DIM)))
		isplit_5 = F.relu(self.ifc3(inputs[:, 5:6, -1]))
		isplit_2_flatten, isplit_3_flatten, isplit_4_flatten = isplit_2.flatten(start_dim=1), isplit_3.flatten(start_dim=1), isplit_4.flatten(start_dim=1)
        # print(split_2_flatten.size(), split_3_flatten.size())
		imerge_net = torch.cat([isplit_0, isplit_1, isplit_2_flatten, isplit_3_flatten, isplit_4_flatten, isplit_5], dim=1)
		i = self.iout_linear(imerge_net)

		return logits, F.log_softmax(i, dim=1), i


		# q = F.relu(self.q1(state))
		# q = F.relu(self.q2(q))

		# i = F.relu(self.i1(state))
		# i = F.relu(self.i2(i))
		# i = self.i3(i)
		# return self.q3(q), F.log_softmax(i, dim=1), i
class discrete_BCQ(object):
	def __init__(
		self, 
		num_actions,
		state_dim,
		device,
		BCQ_threshold=0.3,
		discount=0.99,
		optimizer="Adam",
		optimizer_parameters={},
		polyak_target_update=False,
		target_update_frequency=8e3,
		tau=0.005,
		initial_eps = 1,
		end_eps = 0.001,
		eps_decay_period = 25e4,
		eval_eps=0.001,
	):
	
		self.device = device

		# Determine network type
		self.Q = Pen_Q(state_dim, num_actions) #Conv_Q(state_dim[0], num_actions).to(self.device) if is_atari else FC_Q(state_dim, num_actions).to(self.device)
		self.Q_target = copy.deepcopy(self.Q)
		self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

		self.discount = discount

		# Target update rule
		self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
		self.target_update_frequency = target_update_frequency
		self.tau = tau

		# Decay for eps
		self.initial_eps = initial_eps
		self.end_eps = end_eps
		self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

		# Evaluation hyper-parameters
		self.state_shape = (-1, state_dim)
		self.eval_eps = eval_eps
		self.num_actions = num_actions

		# Threshold for "unlikely" actions
		self.threshold = BCQ_threshold

		# Number of training iterations
		self.iterations = 0


	def select_action(self, state, eval=False):
		# Select action according to policy with probability (1-eps)
		# otherwise, select random action
		if np.random.uniform(0,1) > self.eval_eps:
			with torch.no_grad():
				#print(np.reshape(state,(1,self.state_shape[0],self.state_shape[1])).shape)
		
				state = convert_torch(  np.reshape(np.array(state), (1,6,8))) #.reshape(-1,self.state_shape[0],self.state_shape[1])) #torch.FloatTensor(np.array(state)).reshape(self.state_shape).to(self.device)
				#print(state.shape)
				q, imt, i = self.Q(state)
				imt = imt.exp()
				imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
				# Use large negative number to mask actions from argmax
				return int((imt * q + (1. - imt) * -1e8).argmax(1))
		else:
			return np.random.randint(self.num_actions)


	def train(self, replay_buffer):
		# Sample replay buffer
		np.random.shuffle(replay_buffer)
		state, action, next_state, reward, done  = [],[],[],[],[]
		for i in range(64):
			
			state.append(np.array(replay_buffer[i][0]))
			action.append(np.array(replay_buffer[i][1]))
			next_state.append( np.array(replay_buffer[i][2]))
			reward.append(np.array(replay_buffer[i][3]))
			done.append(np.array(replay_buffer[i][4]))
		#print(np.array(state).shape,np.array(next_state).shape)

		state, action, next_state, reward, done = convert_torch(np.array(state)), convert_torch(np.array(action).reshape(64,-1),dtype=np.int64), convert_torch(np.array(next_state)), convert_torch(np.array(reward).reshape(64,-1)), convert_torch(np.array(done ).reshape(64,-1))

		# Compute the target Q value
		with torch.no_grad():
		
			q, imt, i = self.Q( convert_torch(np.array(next_state)))
			
			imt = imt.exp()
			imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()

			# Use large negative number to mask actions from argmax
			next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

			q, imt, i = self.Q_target(next_state)
			target_Q = reward + done * self.discount * q.gather(1, next_action).reshape(-1, 1)
			#print(q.shape,target_Q.shape,reward.shape)

		# Get current Q estimate
		current_Q, imt, i = self.Q(state)
		current_Q = current_Q.gather(1, action)

		# Compute Q loss
		#print(current_Q.shape,target_Q.shape)
		q_loss = F.smooth_l1_loss(current_Q, target_Q)
		#
		# print(imt.shape,action.shape)
		i_loss = F.nll_loss(imt, action.reshape(-1))

		Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

		# Optimize the Q
		self.Q_optimizer.zero_grad()
		Q_loss.backward()
		self.Q_optimizer.step()

		# Update target network by polyak or full copy every X iterations.
		self.iterations += 1
		self.maybe_update_target()


	def polyak_target_update(self):
		for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
		   target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def copy_target_update(self):
		if self.iterations % self.target_update_frequency == 0:
			 self.Q_target.load_state_dict(self.Q.state_dict())