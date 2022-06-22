from fileinput import filename

import os
import sys
from turtle import done
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import tensorflow.compat.v1 as tf
import load_trace
#import a2c as network
#import d3qn as network
import fixed_env as env
import h5py

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000


TEST_TRACES = './test/'
gamma = 0.99
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward

#name_list =  ['s','a','next_s','r','done']
# name_list = ['s','a','p','g']


class Env():
    def __init__(self, random_seed=RANDOM_SEED):
        np.random.seed(RANDOM_SEED)

        assert len(VIDEO_BIT_RATE) == A_DIM

        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

        # h5py_cnt = 0
        # for s in name_list :
        #     file_name = s+'.h5'
        #     if os.path.exists(file_name):
        #         cmd = 'rm '+file_name
        #         os.system(cmd)
        # net_env = env.Environment(all_cooked_time=all_cooked_time,
        #                           all_cooked_bw=all_cooked_bw,h5py_file,h5py_cnt)
        self.net_env = env.Environment(all_cooked_time=all_cooked_time,
                                all_cooked_bw=all_cooked_bw)

 
  
        self.time_stamp = 0

        self.last_bit_rate = DEFAULT_QUALITY
        self.bit_rate = DEFAULT_QUALITY

        self.s_batch = [np.zeros((S_INFO, S_LEN))]
       
        self.r_batch = []
    
    def reset(self):
        self.last_bit_rate = DEFAULT_QUALITY
        self.bit_rate = DEFAULT_QUALITY  # use the default action here
        del self.s_batch[:]
        tuple_list =[]
        self.s_batch.append(np.zeros((S_INFO, S_LEN)))
        return self.s_batch[0]
 
    def step(self,bit_rate):
        delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                self.net_env.get_video_chunk(bit_rate)

        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smoothness
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                            VIDEO_BIT_RATE[self.last_bit_rate]) / M_IN_K

      

        self.last_bit_rate = bit_rate



        # retrieve previous state
        if len(self.s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(self.s_batch[-1], copy=True)

        # dequeue history record
        current_state = state
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

      
        self.s_batch.append(state)

        return state,reward,done,''

        
       



if __name__ == '__main__':
    main()
