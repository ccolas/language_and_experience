"""
Deep Reinforcement Learning (DQN) Baseline Agent.

This module implements a Deep Q-Network (DQN) agent as a baseline comparison
for the Bayesian theory-learning approach. The agent learns directly from
pixel/state observations without explicit theory formation.

This baseline is described in the paper's experimental comparison section.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import Tuple, Optional
from src.agent.datastore import DataStore
from src.utils import pickle_save
from mpi4py import MPI
from dataclasses import dataclass


# Reuse the layer initialization function
def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# Experience replay buffer
Experience = namedtuple('Experience', ['grid_state', 'internal_state', 'action', 'reward', 'next_grid_state', 'next_internal_state', 'done'])


class ReplayBuffer:
    def __init__(self, capacity: int, device: str):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        experiences = random.sample(self.buffer, batch_size)

        # Separate experiences into batches
        grid_states = torch.stack([exp.grid_state for exp in experiences])
        internal_states = torch.stack([exp.internal_state for exp in experiences])
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float).to(self.device)
        next_grid_states = torch.stack([exp.next_grid_state for exp in experiences])
        next_internal_states = torch.stack([exp.next_internal_state for exp in experiences])
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float).to(self.device)

        return grid_states, internal_states, actions, rewards, next_grid_states, next_internal_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    def __init__(self, grid_size, grid_shape: tuple, internal_state_dim: int, num_actions: int):
        super().__init__()

        kernel_size = 3 * grid_size  # 3 objects wide
        stride = grid_size  # Move by one object
        padding = grid_size  # One object of padding

        # Grid encoder (
        self.grid_encoder = nn.Sequential(
            layer_init(nn.Conv2d(grid_shape[2], 32, kernel_size=kernel_size, stride=stride, padding=padding)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate CNN output size
        with torch.no_grad():
            dummy_grid = torch.zeros((1, grid_shape[2], grid_shape[0], grid_shape[1]))
            cnn_out_size = self.grid_encoder(dummy_grid).shape[1]

        # Internal state processing (same as PPO)
        self.internal_encoder = nn.Sequential(
            layer_init(nn.Linear(internal_state_dim, 64)),
            nn.Tanh(),
        )

        combined_dim = cnn_out_size + 64

        # Q-value network
        self.q_network = nn.Sequential(
            layer_init(nn.Linear(combined_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, num_actions), std=0.1)
        )

    def forward(self, grid, internal_state):
        if len(grid.shape) == 5:
            grid = grid.squeeze(1)
        grid = grid.permute(0, 3, 1, 2)
        grid_features = self.grid_encoder(grid)
        internal_state = internal_state.squeeze(1)  # Remove middle dimension
        internal_features = self.internal_encoder(internal_state)
        combined = torch.cat([grid_features, internal_features], dim=1)
        return self.q_network(combined)


class DQNAgent:
    def __init__(self, game, params):
        self.game = game
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.verbose = params['verbose'] if self.rank == 0 else False
        self.comm_engine = None

        self.params = params
        self.time_tracker = params['time_tracker']
        self.datastore = self.setup_data_store()
        self.device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
        self.history = []
        self.data_path = params['exp_path'] + 'dumps/'

        # Environment setup (similar to PPO)
        formats = self.params['true_game_info']['formats']
        self.max_width = np.max([f[0] for f in formats])
        self.max_height = np.max([f[1] for f in formats])
        self.grid_size = self._compute_grid_size()
        self.objs = list(self.game.rules.obj.keys())

        # Get observation space
        self.i_steps = 0
        obs = self.format_obs(self.game.reset(lvl=0)['state'])
        self.grid_shape = obs['grid'].shape
        self.internal_state_dim = obs['internal_state'].size

        # DQN specific hyperparameters
        self.num_actions = 6

        # DQN hyperparameters
        self.batch_size = 64  # Increased from 32
        self.gamma = 0.95  # Slightly lower for shorter horizon
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 2000  # Faster decay
        self.target_update_freq = 200  # More frequent updates
        self.learning_rate = 0.001  # Higher learning rate

        # Initialize networks
        self.policy_net = DQNNetwork(self.grid_size, self.grid_shape, self.internal_state_dim, self.num_actions).to(self.device)
        self.target_net = DQNNetwork(self.grid_size, self.grid_shape, self.internal_state_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Initialize optimizer and replay buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(capacity=10000, device=self.device)

        # Training metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.current_episode_reward = 0
        self.current_episode_length = 0

        # Add to __init__ after other metrics initialization
        self.debug_frequency = 100  # Log debug info every N steps
        self.q_values_history = deque(maxlen=100)  # Track Q-value statistics
        self.loss_history = deque(maxlen=100)  # Track loss values

    def _compute_grid_size(self) -> int:
        min_speed = 1
        for obj in self.game.rules.obj.dict.values():
            speed = obj.params.get('speed')
            if speed and speed < min_speed:
                min_speed = speed
        return int(1 / min_speed)

    def format_obs(self, state) -> dict:
        grid = np.zeros((self.max_width * self.grid_size, self.max_height * self.grid_size, len(self.objs)))
        agent_pos_normalized = np.array([-1, -1])
        resource = np.array([0])

        for col in state:
            for cell in col:
                for obj in cell:
                    i_obj = self.objs.index(obj['name'])
                    x, y = obj['pos']
                    x_grid = int(x * self.grid_size)
                    y_grid = int(y * self.grid_size)
                    grid[x_grid:x_grid + self.grid_size, y_grid:y_grid + self.grid_size, i_obj] += 1

                    if obj['name'] == 'avatar':
                        agent_pos_normalized = np.array(obj['pos']) / np.array([self.max_width, self.max_height])
                        resources = obj.get('resources', {})
                        if resources:
                            resource = np.array(list(resources.values()))

        internal_state = np.concatenate([resource, agent_pos_normalized])

        return {'grid': grid, 'internal_state': internal_state}

    def select_action(self, grid_state: torch.Tensor, internal_state: torch.Tensor) -> int:
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.i_steps / self.epsilon_decay)

        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.policy_net(grid_state, internal_state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.num_actions)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample transitions from replay buffer
        grid_states, internal_states, actions, rewards, next_grid_states, next_internal_states, dones = \
            self.memory.sample(self.batch_size)

        # Compute current Q values
        current_q_values = self.policy_net(grid_states, internal_states).gather(1, actions.unsqueeze(1))

        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_grid_states, next_internal_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and optimize
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        # Add after computing loss
        self.loss_history.append(loss.item())
        self.q_values_history.append({
            'mean': current_q_values.mean().item(),
            'max': current_q_values.max().item(),
            'min': current_q_values.min().item()
        })

        if self.i_steps % self.debug_frequency == 0:
            recent_loss = np.mean(list(self.loss_history)[-100:])
            recent_q = np.mean([q['mean'] for q in list(self.q_values_history)[-100:]])
            print(f"\nStep {self.i_steps}")
            print(f"Recent loss: {recent_loss:.4f}")
            print(f"Recent Q-values: {recent_q:.4f}")

    def act(self, step_info: dict, transition: dict, end_episode: bool = False) -> Optional[int]:
        if self.rank != 0:
            return None
        self.i_steps += 1
        reward = float(transition['reward'])

        # Handle terminal rewards
        if transition['won']:
            reward += 10
        elif transition['lose']:
            reward -= 10

        # Get current state
        formatted_obs = self.format_obs(transition['state'])
        grid_state = torch.FloatTensor(formatted_obs['grid']).unsqueeze(0).to(self.device)
        internal_state = torch.FloatTensor(formatted_obs['internal_state']).unsqueeze(0).to(self.device)

        # Select and perform action
        action = self.select_action(grid_state, internal_state)

        # Store transition in replay buffer if we have previous state
        if hasattr(self, 'last_grid_state'):
            self.memory.push(Experience(
                self.last_grid_state,
                self.last_internal_state,
                self.last_action,
                reward,
                grid_state,
                internal_state,
                transition['done']
            ))

        # Optimize model
        self.optimize_model()

        # Update target network
        if self.i_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Save current state
        self.last_grid_state = grid_state
        self.last_internal_state = internal_state
        self.last_action = action

        # Update episode stats
        self.current_episode_reward += reward
        self.current_episode_length += 1

        # Handle episode end
        if transition['done']:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            # Log performance stats
            print(f"\nEpisode {len(self.episode_rewards)} stats:")
            print(f"True reward: {self.current_episode_reward:.2f}")
            print(f"Average reward (last 100): {np.mean(list(self.episode_rewards)):.2f}")
            print(f"Average length (last 100): {np.mean(list(self.episode_lengths)):.2f}")
            print(f"Epsilon: {self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.i_steps / self.epsilon_decay):.3f}")

            # Reset episode trackers
            self.current_episode_reward = 0
            self.current_episode_length = 0
            if hasattr(self, 'last_grid_state'):
                delattr(self, 'last_grid_state')
                delattr(self, 'last_internal_state')
                delattr(self, 'last_action')
            return None

        return action

    def reset(self, lvl):
        """Reset environment and agent state"""
        self.lvl = lvl
        # self.i_steps = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        if hasattr(self, 'last_state'):
            if hasattr(self, 'last_grid_state'):
                delattr(self, 'last_grid_state')
                delattr(self, 'last_internal_state')
                delattr(self, 'last_action')

    def store(self, sensorimotor_data=None, linguistic_data=None):
        """Store data in datastore"""
        if self.rank == 0:
            self.time_tracker.tic('main_store_data')
            self.datastore.load(linguistic_data=linguistic_data, sensorimotor_data=sensorimotor_data)
            self.time_tracker.toc('main_store_data')

    def think(self, step_info):
        """Think about current state (placeholder)"""
        pass

    def dump_data(self, life_step_tracker):
        """Dump agent data"""
        if self.rank == 0:
            self.datastore.dump_data(life_step_tracker)
            name = f'thinking_output_generation_{life_step_tracker["gen"]}_life_{life_step_tracker["life"]}_lvl_solved_{life_step_tracker["n_levels_solved"]}.pkl'
            pickle_save(self.history, self.data_path + name)

    def setup_data_store(self):
        return DataStore(self.params) if self.rank == 0 else None
