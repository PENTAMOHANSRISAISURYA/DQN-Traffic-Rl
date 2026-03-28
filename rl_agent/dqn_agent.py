"""
dqn_agent.py  —  v4
====================
Structural fixes for non-converging loss:

  FIX 1 — Soft target network update (Polyak averaging)
    Old: hard copy every 300 steps → target jumps → Q-values chase moving target
    New: target = tau*policy + (1-tau)*target every step (tau=0.005)
         Target shifts smoothly → loss can actually converge

  FIX 2 — Learning rate scheduler
    Old: fixed LR=3e-4 forever → overshoots once loss gets small
    New: CosineAnnealingLR decays LR from 3e-4 → 1e-5 over training
         Network fine-tunes gradually → loss peaks then falls

  FIX 3 — Smaller replay buffer with prioritised sampling
    Old: 20,000 capacity — early random transitions pollute later learning
    New: 10,000 capacity — old bad transitions expire faster

  FIX 4 — Gradient clipping tightened
    Old: max_norm=1.0
    New: max_norm=0.5 — prevents occasional large gradient spikes
         that cause the loss to jump
"""

import os
import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


class DQNNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer     = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, priority=1.0):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max(float(priority), 1e-5))

    def sample(self, batch_size):
        probs   = np.array(self.priorities, dtype=np.float32)
        probs  /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False, p=probs)
        batch = [self.buffer[i] for i in indices]
        s, a, r, ns, d = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.int64),
                np.array(r,  dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d,  dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:

    GAMMA         = 0.97
    LR            = 3e-4
    LR_MIN        = 1e-5
    BATCH_SIZE    = 64
    TAU           = 0.005     # soft update coefficient
    EPSILON_START = 1.0
    EPSILON_MIN   = 0.05
    # Reaches 0.05 by episode 800 of 1000
    EPSILON_DECAY = 0.9963

    def __init__(self, state_size: int, action_size: int,
                 total_steps: int = 200_000, device='auto'):
        self.state_size  = state_size
        self.action_size = action_size
        self.epsilon     = self.EPSILON_START
        self.step_count  = 0

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        ) if device == 'auto' else torch.device(device)

        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR)
        # LR decays smoothly from LR → LR_MIN over total_steps
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=self.LR_MIN
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.memory  = ReplayBuffer(capacity=10_000)

        print(f"[DQNAgent] device={self.device} | "
              f"state={state_size} | actions={action_size} | "
              f"soft-update tau={self.TAU}")

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        self.policy_net.eval()
        with torch.no_grad():
            q = self.policy_net(
                torch.FloatTensor(state).unsqueeze(0).to(self.device))
        self.policy_net.train()
        return int(q.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.policy_net.eval()
        with torch.no_grad():
            s  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            ns = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            qv = self.policy_net(s)[0][action].item()
            qn = self.target_net(ns).max().item()
            td = abs(reward + self.GAMMA * qn * (1 - done) - qv)
        self.policy_net.train()
        self.memory.push(state, action, reward, next_state, done,
                         priority=td + 1e-5)

    def learn(self):
        if len(self.memory) < self.BATCH_SIZE:
            return None

        s, a, r, ns, d = self.memory.sample(self.BATCH_SIZE)

        s_t  = torch.FloatTensor(s).to(self.device)
        a_t  = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r_t  = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns_t = torch.FloatTensor(ns).to(self.device)
        d_t  = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        current_q = self.policy_net(s_t).gather(1, a_t)

        with torch.no_grad():
            best_a   = self.policy_net(ns_t).argmax(1, keepdim=True)
            target_q = self.target_net(ns_t).gather(1, best_a)
            target_q = r_t + self.GAMMA * target_q * (1 - d_t)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.scheduler.step()

        # FIX 1: Soft update — target shifts smoothly every step
        for tp, pp in zip(self.target_net.parameters(),
                          self.policy_net.parameters()):
            tp.data.copy_(self.TAU * pp.data + (1 - self.TAU) * tp.data)

        self.step_count += 1
        return loss.item()

    def decay_epsilon(self):
        """Call once per episode."""
        self.epsilon = max(self.EPSILON_MIN,
                           self.epsilon * self.EPSILON_DECAY)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def save(self, path='results/dqn_traffic.pth'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy':    self.policy_net.state_dict(),
            'target':    self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon':   self.epsilon,
            'steps':     self.step_count,
        }, path)
        print(f"[DQNAgent] Saved -> {path}")

    def load(self, path='results/dqn_traffic.pth'):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy'])
        self.target_net.load_state_dict(ckpt['target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        self.epsilon    = ckpt['epsilon']
        self.step_count = ckpt['steps']
        print(f"[DQNAgent] Loaded <- {path}")