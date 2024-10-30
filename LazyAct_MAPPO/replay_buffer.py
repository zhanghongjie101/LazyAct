import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.device = args.device
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim + 1
        self.action_dim = args.action_dim
        self.episode_limit = args.episode_limit
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.max_episode_len = 0
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {'obs_n': np.zeros([self.batch_size, self.episode_limit, self.N, self.obs_dim]),
                       's': np.zeros([self.batch_size, self.episode_limit, self.state_dim]),
                       'v_n': np.zeros([self.batch_size, self.episode_limit + 1, self.N]),
                       'c_n': np.zeros([self.batch_size, self.episode_limit + 1, self.N]),
                       'avail_a_n': np.ones([self.batch_size, self.episode_limit, self.N, self.action_dim]),  # Note: We use 'np.ones' to initialize 'avail_a_n'
                       'a_n': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'a_logprob_n': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'skip_n': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'skip_logprob_n': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'r': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'c': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'dw': np.ones([self.batch_size, self.episode_limit, self.N]),  # Note: We use 'np.ones' to initialize 'dw'
                       'skip_mask': np.ones([self.batch_size, self.episode_limit, self.N]),
                       'active': np.zeros([self.batch_size, self.episode_limit, self.N])
                       }
        self.episode_num = 0
        self.max_episode_len = 0

    def store_transition(self, episode_step, obs_n, s, v_n, c_n, avail_a_n, a_n, a_logprob_n, skip_n, skip_logprob_n, r, c, dw, skip_mask):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['c_n'][self.episode_num][episode_step] = c_n
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['a_logprob_n'][self.episode_num][episode_step] = a_logprob_n
        self.buffer['skip_n'][self.episode_num][episode_step] = skip_n
        self.buffer['skip_logprob_n'][self.episode_num][episode_step] = skip_logprob_n
        self.buffer['r'][self.episode_num][episode_step] = np.array(r).repeat(self.N)
        self.buffer['c'][self.episode_num][episode_step] = np.array(c).repeat(self.N)
        self.buffer['dw'][self.episode_num][episode_step] = np.array(dw).repeat(self.N)
        self.buffer['active'][self.episode_num][episode_step] = np.ones(self.N)
        self.buffer['skip_mask'][self.episode_num][episode_step] = skip_mask

    def store_last_value(self, episode_step, v_n, c_n):
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['c_n'][self.episode_num][episode_step] = c_n
        self.episode_num += 1
        # Record max_episode_len
        if episode_step > self.max_episode_len:
            self.max_episode_len = episode_step

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'a_n' or key == 'skip_n':
                batch[key] = torch.tensor(self.buffer[key][:, :self.max_episode_len], dtype=torch.long).to(self.device)
            elif key == 'v_n' or key == 'c_n':
                batch[key] = torch.tensor(self.buffer[key][:, :self.max_episode_len + 1], dtype=torch.float32).to(self.device)
            else:
                batch[key] = torch.tensor(self.buffer[key][:, :self.max_episode_len], dtype=torch.float32).to(self.device)
        return batch
