import os, sys

import torch
import time
import numpy as np
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim import RMSprop, Adam
import tensorflow as tf
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Learner(object):
    def __init__(self, args, q_batch, actor_critic, model_filepath, train_steps):
        self.args = args
        self.q_batch = q_batch
        self.actor_critic = actor_critic
        self.train_steps = train_steps
        self.model_filepath = model_filepath
        self.lr_annealing_steps = args.lr_annealing_steps
        self.initial_lr = args.lr
        self.lam_lr = args.lam_lr
        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.initial_lr, betas=(0.9, 0.999), eps=1e-8)

    def get_lr(self):
        if self.train_steps.value <= self.lr_annealing_steps:
            return self.initial_lr - (self.train_steps.value * self.initial_lr / self.lr_annealing_steps)
        else:
            return 0.0

    def learning(self):
        self.summary_writer = tf.summary.FileWriter(os.path.join(self.args.debugging_folder, self.args.game + '/learner'))
        torch.manual_seed(self.args.seed)
        coef_hat = torch.Tensor([self.args.coef_hat]).to(device)
        rho_hat = torch.Tensor([self.args.rho_hat]).to(device)
        lam = torch.FloatTensor([self.args.init_lam]).to(device)
        torch.save(self.actor_critic, self.model_filepath)
        while self.train_steps.value<=self.args.total_num_steps:
            obs, actions, skips_pad, rewards, reward_costs, log_probs, log_skips, masks, skip_ratio = self.q_batch.get(block=True)
            obs_all = obs.view((-1,4,84,84))
            actions_all = actions.view((-1,1))
            skips_all_pad = skips_pad.view(-1)
            skips = skips_pad[:,:-1,:]
            skips_all = skips.reshape((-1,1))
            log_probs_all = log_probs.view((-1,1)) + log_skips.view((-1,1))
            #current model predict
            #predict skip
            skip_policies, hiddens = self.actor_critic(obs_all)
            #predict action
            skips_onehot_all = torch.zeros([obs_all.size(0), self.actor_critic.num_skips]).to(device)
            skips_onehot_all[torch.arange(obs_all.size(0)).long(), skips_all_pad] = 1.0
            policies, value, cost = self.actor_critic(hiddens, skips_onehot_all.detach())
            #value insert 0
            value_skip = torch.zeros([skips.size(0),skips.size(1)+1,skips.size(2)]).to(device)
            value_skip[:,1:,:] = skips
            value_skip_all = value_skip.view((-1,1))
            value = value * torch.pow(self.args.gamma, value_skip_all)
            #cost insert 0
            cost_skip = torch.zeros([skips.size(0),skips.size(1)+1,skips.size(2)]).to(device)
            cost_skip[:,1:,:] = skips
            cost_skip_all = cost_skip.view((-1,1))
            cost = cost * torch.pow(self.args.gamma, cost_skip_all)
            #action del last
            policies = policies.view((obs.size(0),obs.size(1),policies.size(1)))[:,:-1,:]
            policies = policies.contiguous().view((-1,policies.size(2)))
            network_output_pi = F.softmax(policies, dim=1)
            log_policy = F.log_softmax(policies, dim=1)
            action_log_prob = log_policy[np.arange(log_policy.size(0)), actions_all.view(-1)].view((-1,1))
            #skip del last
            skip_policies = skip_policies.view((obs.size(0),obs.size(1),skip_policies.size(1)))[:,:-1,:]
            skip_policies = skip_policies.contiguous().view((-1,skip_policies.size(2)))
            network_output_skip = F.softmax(skip_policies, dim=-1)
            log_skip = F.log_softmax(skip_policies, dim=-1)
            skip_log_prob = log_skip[np.arange(log_skip.size(0)), skips_all.view(-1)].view((-1,1))
            
            all_log_probs = action_log_prob + skip_log_prob
            # rate action
            is_rate = torch.exp(all_log_probs.detach() - log_probs_all)
            coef = torch.min(coef_hat, is_rate).view((obs.size(0),obs.size(1)-1,1)).permute(1,0,2)
            rho = torch.min(rho_hat, is_rate).view((obs.size(0),obs.size(1)-1,1)).permute(1,0,2)
            
            entropy = (-network_output_pi*log_policy).sum(-1).mean() + (-network_output_skip*skip_log_prob).sum(-1).mean()
            
            values = value.view((obs.size(0),obs.size(1),1)).permute(1,0,2)
            costs = cost.view((obs.size(0),obs.size(1),1)).permute(1,0,2)
            
            log_prob = all_log_probs.view((obs.size(0),obs.size(1)-1,1)).permute(1,0,2)

            policy_loss = []
            vs = torch.zeros((obs.size(1), obs.size(0), 1)).to(device)
            vs[-1] = values[-1]
            cs = torch.zeros((obs.size(1), obs.size(0), 1)).to(device)
            cs[-1] = costs[-1]

            """
            vs: v-trace target
            """
            for rev_step in reversed(range(obs.size(1)-1)):
                # r + args * v(s+1) - V(s)
                delta_s = rho[rev_step] * (rewards[:, rev_step] + self.args.gamma * masks[:, rev_step] \
                                           * values[rev_step+1]-values[rev_step])
                delta_c = rho[rev_step] * (reward_costs[:, rev_step] + self.args.gamma * masks[:, rev_step] \
                                           * costs[rev_step+1]-costs[rev_step])
                # value_loss = v_{s} - V(x_{s})
                advantages = rho[rev_step] * (rewards[:, rev_step] - lam * reward_costs[:, rev_step] + self.args.gamma \
                                * masks[:, rev_step] * (vs[rev_step+1] - lam * cs[rev_step+1]) - (values[rev_step]-lam*costs[rev_step]))
                vs[rev_step] = values[rev_step] + delta_s + self.args.gamma * coef[rev_step] * (vs[rev_step+1]-values[rev_step+1])
                cs[rev_step] = costs[rev_step] + delta_c + self.args.gamma * coef[rev_step] * (cs[rev_step+1]-costs[rev_step+1])
                policy_loss.append(-log_prob[rev_step]*advantages.detach())

            baseline_loss = torch.mean(0.5*(vs[:-1].detach() - values[:-1])**2)
            cost_loss = torch.mean(0.5*(cs[:-1].detach() - costs[:-1])**2)
            entropy_loss = self.args.entropy_coef * entropy
            policy_loss = torch.stack(policy_loss).mean()
            loss = policy_loss + self.args.value_loss_coef * (baseline_loss + cost_loss) - entropy_loss

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            
            if self.train_steps.value % 100 == 0:
                lam = lam + self.lam_lr * (self.args.skip_ratio - skip_ratio.item())
                lam = torch.clamp(lam, 0.0)
            
            print("learner: lambda {:.3f} v_loss {:.3f} c loss {:.3f} p_loss {:.3f} entropy_loss {:.5f} loss {:.3f}".format(lam.item(), \
                baseline_loss.item(), cost_loss.item(), policy_loss.item(), entropy_loss.item(), loss.item()))
            summary = tf.Summary(value=[tf.Summary.Value(tag='learner/total_loss', simple_value=float(loss.item())),
                                        tf.Summary.Value(tag='learner/lambda', simple_value=float(lam.item())),
                                        tf.Summary.Value(tag='learner/skip_ratio', simple_value=float(skip_ratio.item())),
                                        tf.Summary.Value(tag='learner/value_loss', simple_value=float(baseline_loss.item())),
                                        tf.Summary.Value(tag='learner/cost_loss', simple_value=float(cost_loss.item())),])
            self.summary_writer.add_summary(summary, self.train_steps.value)
            self.summary_writer.flush()

            if (self.train_steps.value % 10 == 0):
                torch.save(self.actor_critic, self.model_filepath)
            self.train_steps.value = self.train_steps.value + 1