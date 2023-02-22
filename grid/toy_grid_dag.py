import argparse
import copy
import gzip
import heapq
import itertools
import os
import pickle
from collections import defaultdict
from itertools import count

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

import random
import sys
import tempfile
import datetime

from itertools import chain

parser = argparse.ArgumentParser()

parser.add_argument("--device", default='cuda', type=str)
parser.add_argument("--progress", action='store_true')
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--tb_lr", default=0.001, help="Learning rate", type=float)
parser.add_argument("--tb_z_lr", default=0.1, help="Learning rate", type=float)
parser.add_argument("--mbsize", default=16, help="Minibatch size", type=int)
parser.add_argument("--n_hid", default=256, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--n_train_steps", default=20000, type=int)
parser.add_argument("--horizon", default=8, type=int)
parser.add_argument("--ndim", default=2, type=int)
parser.add_argument("--augmented", default=0, type=int)
parser.add_argument("--ri_eta", default=0.001, type=float)

_dev = [torch.device('cuda')]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])

def set_device(dev):
    _dev[0] = dev


def func_corners(x, horizon, kind=None):
    r = (x == horizon - 2).prod(-1) + (x[0] == horizon - 2) * (x[1] == 1) + (x[1] == horizon - 2) * (x[0] == 1)
    return r


class GridEnv:
    def __init__(self, horizon, ndim=2, xrange=[-1, 1], func=None):
        assert args.ndim == 2

        self.horizon = horizon
        self.start = [xrange[0]] * ndim
        self.ndim = ndim
        self.func = func
        self.xspace = np.linspace(*xrange, horizon)

        self._true_density = None

        rs = []
        for i in range(self.horizon):
            for j in range(self.horizon):
                rs.append(self.func(np.int32([i, j]), horizon=self.horizon))
        rs = np.array(rs)

        self.true_density = rs / rs.sum()
        self.goals = [[self.horizon - 2, self.horizon - 2], [self.horizon - 2, 1], [1, self.horizon - 2]]

    def obs(self, s=None):
        s = np.int32(self._state if s is None else s)
        z = np.zeros((self.horizon * self.ndim), dtype=np.float32)
        z[np.arange(len(s)) * self.horizon + s] = 1
        return z

    def s2x(self, s):
        x = (self.obs(s).reshape((self.ndim, self.horizon)) * self.xspace[None, :]).sum(1)
        return x

    def reset(self):
        self._state = np.int32([0] * self.ndim)
        self._step = 0
        rew = self.func(self._state, horizon=self.horizon)
        return self.obs(), rew, self._state

    def step(self, a, s=None):
        _s = s
        s = (self._state if s is None else s) + 0
        
        if a < self.ndim:
            s[a] += 1

        done = s.max() >= self.horizon - 1 or a == self.ndim

        if _s is None:
            self._state = s
            self._step += 1

        rew = self.func(s, horizon=self.horizon) if done else 0

        return self.obs(s), rew, done, s

    def true_density(self):
        if self._true_density is not None:
            return self._true_density

        all_int_states = np.int32(list(itertools.product(*[list(range(self.horizon))] * self.ndim)))
        state_mask = np.array([len(self.parent_transitions(s, False)[0]) > 0 or sum(s) == 0 for s in all_int_states])

        traj_rewards = self.func(all_int_states, horizon=self.horizon)[state_mask]

        self._true_density = (traj_rewards / traj_rewards.sum(), list(map(tuple, all_int_states[state_mask])), traj_rewards)
        return self._true_density


def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
    return nn.Sequential(*(sum([[nn.Linear(i, o)] + ([act] if n < len(l)-2 else []) for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))


class RND(nn.Module):
    def __init__(self, state_dim, reward_scale=0.5, hidden_dim=256, s_latent_dim=128):
        super(RND, self).__init__()

        self.random_target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, s_latent_dim)
        )

        self.predictor_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, s_latent_dim),
        )
        
        self.reward_scale = reward_scale

    def forward(self, next_state):
        random_phi_s_next = self.random_target_network(next_state)
        predicted_phi_s_next = self.predictor_network(next_state)
        return random_phi_s_next, predicted_phi_s_next

    def compute_intrinsic_reward(self, next_states):
        random_phi_s_next, predicted_phi_s_next = self.forward(next_states)

        intrinsic_reward = torch.norm(predicted_phi_s_next.detach() - random_phi_s_next.detach(), dim=-1, p=2)
        intrinsic_reward *= self.reward_scale

        intrinsic_reward = intrinsic_reward.cpu().detach().numpy()

        return intrinsic_reward

    def compute_loss(self, next_states):
        random_phi_s_next, predicted_phi_s_next = self.forward(next_states)
        rnd_loss = torch.norm(predicted_phi_s_next - random_phi_s_next.detach(), dim=-1, p=2)
        mean_rnd_loss = torch.mean(rnd_loss)
        return mean_rnd_loss


class TBFlowNetAgent:
    def __init__(self, args, envs):
        out_dim = 2 * args.ndim + 1
        if args.augmented:
            out_dim += 1
        self.model = make_mlp([args.horizon * args.ndim] + [args.n_hid] * args.n_layers + [out_dim])
        self.model.to(args.dev)

        self.augmented = args.augmented
        if self.augmented:
            self.intrinsic_reward_model = RND(args.horizon * args.ndim, args.ri_eta)
            self.intrinsic_reward_model.to(args.dev)
            self.ri_loss_coe = 1.

        self.Z = torch.zeros((1,)).to(args.dev)
        self.Z.requires_grad_()

        self.dev = args.dev

        self.envs = envs
        self.ndim = args.ndim
        self.horizon = args.horizon
            
        self.goal_found_map = {}
        for goal in self.envs[0].goals:
            self.goal_found_map[str(goal)] = 0

    def parameters(self):
        if self.augmented:
            return chain(self.model.parameters(), self.intrinsic_reward_model.parameters())
        else:
            return self.model.parameters()

    def sample_many(self, mbsize, all_visited, to_print=False):
        inf = 1000000000

        batch_s, batch_a, batch_next_s, batch_ri = [[] for i in range(mbsize)], [[] for i in range(mbsize)], [[] for i in range(mbsize)], [[] for i in range(mbsize)]
        env_idx_done_map = {i: False for i in range(mbsize)}
        not_done_envs = [i for i in range(mbsize)]
        env_idx_return_map = {}

        s = tf([i.reset()[0] for i in self.envs])
        done = [False] * mbsize
        
        while not all(done):
            with torch.no_grad():
                pred = self.model(s)

                z = torch.where(s > 0)[1].reshape(s.shape[0], -1)
                z[:, 1] -= self.horizon

                edge_mask = torch.cat([(z == self.horizon - 1).float(), torch.zeros((len(done) - sum(done), 1), device=self.dev)], 1)
                logits = (pred[..., : self.ndim + 1] - inf * edge_mask).log_softmax(1)

                sample_ins_probs = logits.softmax(1)
                acts = sample_ins_probs.multinomial(1).squeeze(-1)

            step = [i.step(a) for i, a in zip([e for d, e in zip(done, self.envs) if not d], acts)]

            if self.augmented:
                next_s = tf([i[0] for i in step])
                intrinsic_rewards = self.intrinsic_reward_model.compute_intrinsic_reward(next_s)

            for dat_idx, (curr_s, curr_a) in enumerate(zip(s, acts)):
                env_idx = not_done_envs[dat_idx]

                curr_formatted_s = torch.where(curr_s > 0)[0]
                curr_formatted_s[1] -= self.horizon

                batch_s[env_idx].append(curr_formatted_s)
                batch_a[env_idx].append(curr_a.unsqueeze(-1))

                if self.augmented:
                    batch_next_s[env_idx].append(next_s[dat_idx])
                    batch_ri[env_idx].append(intrinsic_rewards[dat_idx])

            for dat_idx, (ns, r, d, _) in enumerate(step):
                env_idx = not_done_envs[dat_idx]
                env_idx_done_map[env_idx] = d.item()

                if d.item():
                    env_idx_return_map[env_idx] = r.item()

                    formatted_ns = np.where(ns > 0)[0]
                    formatted_ns[1] -= self.horizon

                    batch_s[env_idx].append(tl(formatted_ns.tolist()))

            not_done_envs = []
            for env_idx in env_idx_done_map:
                if not env_idx_done_map[env_idx]:
                    not_done_envs.append(env_idx)

            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf([i[0] for i in step if not i[2]])

            for (_, r, d, sp) in step:
                if d:
                    all_visited.append(sp[0] * self.horizon + sp[1])
                    
        batch_steps = [len(batch_s[i]) for i in range(len(batch_s))]

        for i in range(len(batch_s)):
            batch_s[i] = torch.stack(batch_s[i])
            batch_a[i] = torch.stack(batch_a[i])

            assert batch_s[i].shape[0] - batch_a[i].shape[0] == 1
            if self.augmented:
                batch_next_s[i] = torch.stack(batch_next_s[i])
                batch_ri[i] = torch.tensor(batch_ri[i]).unsqueeze(-1).float().to(self.dev)

        batch_R = [env_idx_return_map[i] + batch_ri[i][-1].item() if self.augmented else env_idx_return_map[i] for i in range(len(batch_s))]

        for i in range(len(batch_s)):
            curr_s = str(batch_s[i][-1].data.cpu().numpy().tolist())
            if curr_s in self.goal_found_map:
                self.goal_found_map[curr_s] += 1

        modes_cnt = 0
        for goal in self.goal_found_map:
            if self.goal_found_map[goal] > 0:
                modes_cnt += 1

        return [batch_s, batch_a, batch_R, batch_steps, batch_next_s, batch_ri]

    def convert_states_to_onehot(self, states):
        return torch.nn.functional.one_hot(states, self.horizon).view(states.shape[0], -1).float()

    def learn_from(self, it, batch):
        inf = 1000000000

        states, actions, returns, episode_lens, next_states, intrinsic_rewards = batch
        returns = torch.tensor(returns).to(self.dev)

        ll_diff = []
        for data_idx in range(len(states)):
            curr_episode_len = episode_lens[data_idx]

            curr_states = states[data_idx][:curr_episode_len, :]
            curr_actions = actions[data_idx][:curr_episode_len - 1, :]
            curr_return = returns[data_idx]

            curr_states_onehot = self.convert_states_to_onehot(curr_states)

            pred = self.model(curr_states_onehot)

            edge_mask = torch.cat([(curr_states == self.horizon - 1).float(), torch.zeros((curr_states.shape[0], 1), device=self.dev)], 1)
            logits = (pred[..., :self.ndim + 1] - inf * edge_mask).log_softmax(1) 

            init_edge_mask = (curr_states == 0).float()
            back_logits_end_pos = -1 if self.augmented else pred.shape[-1]
            back_logits = (pred[..., self.ndim + 1:back_logits_end_pos] - inf * init_edge_mask).log_softmax(1)

            logits = logits[:-1, :].gather(1, curr_actions).squeeze(1) 
            back_logits = back_logits[1:-1, :].gather(1, curr_actions[:-1, :]).squeeze(1) if curr_actions[-1] == self.ndim else back_logits[1:, :].gather(1, curr_actions).squeeze(1)

            sum_logits = torch.sum(logits)
            if self.augmented:
                curr_intrinsic_rewards = intrinsic_rewards[data_idx].squeeze(-1)[:-1] 
                flow = (pred[..., -1][1:-1]).exp()
                augmented_r_f = curr_intrinsic_rewards / flow    
                sum_back_logits = torch.sum((back_logits.exp() + augmented_r_f).log()) if curr_actions[-1] == self.ndim else torch.sum((back_logits[:-1].exp() + augmented_r_f).log()) + back_logits[-1]
            else:
                sum_back_logits = torch.sum(back_logits)

            curr_return = curr_return.float() + 1e-8

            curr_ll_diff = self.Z + sum_logits - curr_return.log() - sum_back_logits
            ll_diff.append(curr_ll_diff ** 2)

        loss = torch.cat(ll_diff).sum() / len(states)
        if self.augmented:
            rnd_loss = torch.stack([self.intrinsic_reward_model.compute_loss(next_states[data_idx]) for data_idx in range(len(states))]).sum() / len(states)
            loss += self.ri_loss_coe * rnd_loss

        return loss


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    args.dev = torch.device(args.device)
    set_device(args.dev)

    envs = [GridEnv(args.horizon, args.ndim, func=func_corners) for i in range(args.mbsize)]

    agent = TBFlowNetAgent(args, envs)
    opt = torch.optim.Adam([{'params': agent.parameters(), 'lr': args.tb_lr}, {'params':[agent.Z], 'lr': args.tb_z_lr}])

    all_visited = []
    for i in tqdm(range(args.n_train_steps + 1), disable=not args.progress):
        data = agent.sample_many(args.mbsize, all_visited)

        loss = agent.learn_from(i, data)
        loss.backward()
        opt.step()
        opt.zero_grad()

        if i % 10 == 0:
            emp_dist = np.bincount(all_visited, minlength=len(envs[0].true_density)).astype(float)
            emp_dist /= emp_dist.sum()
            l1 = np.abs(envs[0].true_density - emp_dist).mean()

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_num_threads(1)
    
    if args.augmented:
        ri_eta_map = {8: 0.005, 16: 0.005, 32: 0.001, 64: 0.005, 128: 0.001}
        args.ri_eta = ri_eta_map[args.horizon]

    main(args)