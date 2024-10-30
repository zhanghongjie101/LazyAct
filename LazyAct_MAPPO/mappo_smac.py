import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import *
import numpy as np
import copy


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc_skip = nn.Linear(args.rnn_hidden_dim, args.skip_dim)
        self.fc_pi = nn.Linear(args.rnn_hidden_dim + args.skip_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc_skip, gain=0.01)
            orthogonal_init(self.fc_pi, gain=0.01)

    def forward(self, actor_input, avail_a_n, skip=None):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        if skip is None:
            x = self.activate_func(self.fc1(actor_input))
            self.rnn_hidden = self.rnn(x, self.rnn_hidden)
            skip = self.fc_skip(self.rnn_hidden)
            prob = torch.softmax(skip, dim=-1)
            return prob, self.rnn_hidden
        else:
            pi_x = torch.cat((actor_input, skip),-1)
            pi = self.fc_pi(pi_x)
            pi[avail_a_n == 0] = -1e10  # Mask the unavailable actions
            prob = torch.softmax(pi, dim=-1)
            return prob

class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.fc4 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc4)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        cost = self.fc4(self.rnn_hidden)
        return value, cost


class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc_skip = nn.Linear(args.mlp_hidden_dim, args.skip_dim)
        self.fc_pi = nn.Linear(args.mlp_hidden_dim + args.skip_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc_skip, gain=0.01)
            orthogonal_init(self.fc_pi, gain=0.01)
    
    def forward(self, actor_input, avail_a_n, skip=None):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        if skip is None:
            x = self.activate_func(self.fc1(actor_input))
            x = self.activate_func(self.fc2(x))
            skip = self.fc_skip(x)
            prob = torch.softmax(skip, dim=-1)
            return prob, x
        else:
            pi_x = torch.cat((actor_input, skip),-1)
            pi = self.fc_pi(pi_x)
            pi[avail_a_n == 0] = -1e10  # Mask the unavailable actions
            prob = torch.softmax(pi, dim=-1)
            return prob

class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.fc4 = nn.Linear(args.mlp_hidden_dim, 1)
        
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)
            

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, max_episode_len, N, critic_input_dim), value.shape=(mini_batch_size, max_episode_len, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        cost = self.fc4(x)
        return value, cost


class MAPPO_SMAC:
    def __init__(self, args):
        self.N = args.N
        self.device = args.device
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.skip_dim = args.skip_dim
        self.skip_train = args.skip_train
        self.alpha = args.init_alpha
        self.alpha_lr = args.alpha_lr
        self.skip_ratio = args.skip_ratio

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_agent_specific = args.use_agent_specific
        self.use_value_clip = args.use_value_clip
        # get the input dimension of actor and critic
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim + 1
        if self.add_agent_id:
            print("------add agent id------")
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N
        if self.use_agent_specific:
            print("------use agent specific global state------")
            self.critic_input_dim += args.obs_dim
        print('------------------------')
        print(self.actor_input_dim)
        print(self.critic_input_dim)
        print('------------------------')
        if self.use_rnn:
            print("------use rnn------")
            self.actor = Actor_RNN(args, self.actor_input_dim).to(self.device)
            self.critic = Critic_RNN(args, self.critic_input_dim).to(self.device)
        else:
            self.actor = Actor_MLP(args, self.actor_input_dim).to(self.device)
            self.critic = Critic_MLP(args, self.critic_input_dim).to(self.device)

        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.set_adam_eps:
            print("------set adam eps------")
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)

    def choose_action(self, obs_n, avail_a_n, evaluate):
        with torch.no_grad():
            actor_inputs = []
            obs_n = torch.tensor(obs_n, dtype=torch.float32).to(self.device)  # obs_n.shape=(N，obs_dim)
            actor_inputs.append(obs_n)
            if self.add_agent_id:
                """
                    Add an one-hot vector to represent the agent_id
                    For example, if N=3:
                    [obs of agent_1]+[1,0,0]
                    [obs of agent_2]+[0,1,0]
                    [obs of agent_3]+[0,0,1]
                    So, we need to concatenate a N*N unit matrix(torch.eye(N))
                """
                actor_inputs.append(torch.eye(self.N).to(self.device))

            actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_input.shape=(N, actor_input_dim)
            avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32).to(self.device)  # avail_a_n.shape=(N, action_dim)
            skip_prob_zeros = torch.zeros(actor_inputs.shape[0], self.skip_dim).to(self.device)
            skip_prob_zeros[:,0] = 1.0
            if evaluate:  # When evaluating the policy, we select the action with the highest probability
                skip_prob, hidden = self.actor(actor_inputs, avail_a_n)  # skip_prob.shape=(N, skip_dim)
                skip_prob = skip_prob if self.skip_train else skip_prob_zeros
                skip_n = skip_prob.argmax(dim=-1) # skip_n.shape=(N, )
                skip_onehot = skip_n.unsqueeze(1)
                skip_onehot = torch.zeros(skip_n.shape[0], self.skip_dim).to(self.device).scatter_(1,skip_onehot,1) # skip_onehot.shape=(N, skip_dim)
                action_prob = self.actor(hidden, avail_a_n, skip_onehot)  # action_prob.shape=(N, action_dim)
                a_n = action_prob.argmax(dim=-1) # a_n.shape=(N, )
                return skip_n.data.cpu().numpy(), None, a_n.data.cpu().numpy(), None
            else:
                skip_prob, hidden = self.actor(actor_inputs, avail_a_n)  # skip_prob.shape=(N, skip_dim)
                skip_prob = skip_prob if self.skip_train else skip_prob_zeros
                dist_skip = Categorical(probs=skip_prob)
                skip_n = dist_skip.sample()
                skip_logprob_n = dist_skip.log_prob(skip_n)
                skip_onehot = skip_n.unsqueeze(1)
                skip_onehot = torch.zeros(skip_n.shape[0], self.skip_dim).to(self.device).scatter_(1,skip_onehot,1) # skip_onehot.shape=(N, skip_dim)
                action_prob = self.actor(hidden, avail_a_n, skip_onehot)  # action_prob.shape=(N, action_dim)
                dist_action = Categorical(probs=action_prob)
                a_n = dist_action.sample()
                a_logprob_n = dist_action.log_prob(a_n)
                return skip_n.data.cpu().numpy(), skip_logprob_n.data.cpu().numpy(), a_n.data.cpu().numpy(), a_logprob_n.data.cpu().numpy()

    def get_value(self, s, obs_n):
        with torch.no_grad():
            critic_inputs = []
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1).to(self.device)  # (state_dim,)-->(N,state_dim)
            critic_inputs.append(s)
            if self.use_agent_specific:  # Add local obs of agents
                critic_inputs.append(torch.tensor(obs_n, dtype=torch.float32).to(self.device))
            if self.add_agent_id:  # Add an one-hot vector to represent the agent_id
                critic_inputs.append(torch.eye(self.N).to(self.device))
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_input.shape=(N, critic_input_dim)
            v_n, c_n = self.critic(critic_inputs)  # v_n.shape(N,1)
            return v_n.data.cpu().numpy().flatten(), c_n.data.cpu().numpy().flatten()

    def train_alpha(self, skip_ratio):
        self.alpha = self.alpha + self.alpha_lr * (self.skip_ratio - skip_ratio)
        self.alpha = 0.0 if self.alpha < 0.0 else self.alpha
    
    def train(self, replay_buffer, total_steps):
        skip_coef = 1.0 if self.skip_train else 0.0
        batch = replay_buffer.get_training_data()  # Get training data
        max_episode_len = replay_buffer.max_episode_len

        # Calculate the advantage using GAE
        adv = []
        adv_r = []
        adv_c = []
        gae = 0
        gae_r = 0
        gae_c = 0
        with torch.no_grad():  # adv and v_target have no gradient
            # deltas.shape=(batch_size,max_episode_len,N)
            deltas_r = batch['r'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['dw']) - batch['v_n'][:, :-1]
            deltas_c = batch['c'] + self.gamma * batch['c_n'][:, 1:] * (1 - batch['dw']) - batch['c_n'][:, :-1]
            deltas = deltas_r + self.alpha * deltas_c * skip_coef
            for t in reversed(range(max_episode_len)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
                gae_r = deltas_r[:, t] + self.gamma * self.lamda * gae_r
                adv_r.insert(0, gae_r)
                gae_c = deltas_c[:, t] + self.gamma * self.lamda * gae_c
                adv_c.insert(0, gae_c)
            adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,max_episode_len,N)
            adv_r = torch.stack(adv_r, dim=1)  # adv_r.shape(batch_size,max_episode_len,N)
            adv_c = torch.stack(adv_c, dim=1)  # adv_c.shape(batch_size,max_episode_len,N)
            v_target = adv_r + batch['v_n'][:, :-1]  # v_target.shape(batch_size,max_episode_len,N)
            c_target = adv_c + batch['c_n'][:, :-1]  # c_target.shape(batch_size,max_episode_len,N)
            if self.use_adv_norm:  # Trick 1: advantage normalization
                adv_copy = copy.deepcopy(adv.data.cpu().numpy())
                adv_copy[batch['active'].data.cpu().numpy() == 0] = np.nan
                adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5)) #??

        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        actor_inputs, critic_inputs = self.get_inputs(batch, max_episode_len)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                """
                    Get probs_now and values_now
                    probs_now.shape=(mini_batch_size, max_episode_len, N, action_dim)
                    values_now.shape=(mini_batch_size, max_episode_len, N)
                """
                if self.use_rnn:
                    pass
                else:
                    skip_probs_now, hiddens = self.actor(actor_inputs[index], batch['avail_a_n'][index])
                    skip_onehot = batch['skip_n'][index].unsqueeze(-1)
                    skip_onehot = torch.zeros(self.mini_batch_size, max_episode_len, self.N, self.skip_dim).to(self.device).scatter_(3,skip_onehot,1) # skip_onehot.shape=(N, skip_dim)
                    action_probs_now = self.actor(hiddens, batch['avail_a_n'][index], skip_onehot)
                    values_now, costs_now = self.critic(critic_inputs[index])
                    values_now, costs_now = values_now.squeeze(-1), costs_now.squeeze(-1)

                skip_dist_now = Categorical(skip_probs_now)
                skip_dist_entropy = skip_dist_now.entropy()  # dist_entropy.shape=(mini_batch_size, max_episode_len, N)
                # batch['a_n'][index].shape=(mini_batch_size, max_episode_len, N)
                skip_logprob_n_now = skip_dist_now.log_prob(batch['skip_n'][index])  # a_logprob_n_now.shape=(mini_batch_size, max_episode_len, N)
                a_dist_now = Categorical(action_probs_now)
                a_dist_entropy = a_dist_now.entropy()  # dist_entropy.shape=(mini_batch_size, max_episode_len, N)
                # batch['a_n'][index].shape=(mini_batch_size, max_episode_len, N)
                a_logprob_n_now = a_dist_now.log_prob(batch['a_n'][index])  # a_logprob_n_now.shape=(mini_batch_size, max_episode_len, N)
                
                logprob_n_now = skip_logprob_n_now * skip_coef + a_logprob_n_now
                batch_logprob_n = batch['skip_logprob_n'][index].detach() * skip_coef + batch['a_logprob_n'][index].detach()
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(logprob_n_now - batch_logprob_n)  # ratios.shape=(mini_batch_size, max_episode_len, N)
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * (skip_dist_entropy * skip_coef + a_dist_entropy)
                actor_loss = actor_loss * (1.0-batch['skip_mask'][index]) # detach skipped action
                mask_tensor = batch['active'][index] * (1.0-batch['skip_mask'][index])
                actor_loss = (actor_loss * mask_tensor).sum() / mask_tensor.sum()

                if self.use_value_clip:
                    '''
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                    '''
                    pass
                else:
                    critic_value_loss = (values_now - v_target[index]) ** 2
                    critic_cost_loss = (costs_now - c_target[index]) ** 2
                    critic_loss = critic_value_loss + critic_cost_loss * skip_coef
                critic_loss = (critic_loss * mask_tensor).sum() / mask_tensor.sum()

                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss + critic_loss
                ac_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch, max_episode_len):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['obs_n'])
        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
        if self.use_agent_specific:
            critic_inputs.append(batch['obs_n'])
        if self.add_agent_id:
            # agent_id_one_hot.shape=(mini_batch_size, max_episode_len, N, N)
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, max_episode_len, 1, 1).to(self.device)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        return actor_inputs, critic_inputs

    def save_model(self, env_name, number):
        #torch.save(self.actor.state_dict(), "./model/MAPPO_env_{}_actor_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, int(total_steps / 1000)))
        torch.save(self.actor.state_dict(), "./model/MAPPO_env_{}_actor_number_{}.pth".format(env_name, number))

    def load_model(self, env_name, number):
        #self.actor.load_state_dict(torch.load("./model/MAPPO_env_{}_actor_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))
        self.actor.load_state_dict(torch.load("./model/MAPPO_env_{}_actor_number_{}.pth".format(env_name, number), map_location=self.device))
