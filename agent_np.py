import cupy as cp
import json
import matplotlib.pyplot as plt
import torch
import logging

class RacetrackAgent:
    """
    Optimized Q-learning agent with an epsilon-greedy exploration strategy.
    """
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=0.5, mini_epsilon=0.05, decay=0.995):
        """
        Initializes the BaseAgent.
        """
        self.x_max, self.y_max = 18, 32
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.mini_epsilon = mini_epsilon
        self.decay = decay
        
        self.action_space = cp.array([
            (0, 0), (0, 1), (1, 0), (1, 1), (-1, 0), (0, -1), (-1, -1), (1, -1), (-1, 1)
        ], dtype=cp.int64)
        
        self.num_actions = len(self.action_space)
        self.counts = cp.zeros((self.x_max, self.y_max, 5, 5, self.num_actions), dtype=cp.int32)
        self.q_function = cp.zeros((self.x_max, self.y_max, 5, 5, self.num_actions), dtype=cp.float32)
        self.returns = cp.zeros((self.x_max, self.y_max, 5, 5, self.num_actions), dtype=cp.float32)
        self.policy = cp.random.randint(0, self.num_actions, (self.x_max, self.y_max, 5, 5), dtype=cp.int64)
    
    def learning(self, traj):
        """
        Perform Monte Carlo update using batch processing.
        """
        states, actions, rewards = zip(*traj)
        states = cp.array(states)
        actions = cp.array(actions)
        rewards = cp.array(rewards, dtype=cp.float32)
        seq_len = len(rewards)
        
        discount_factors = cp.power(self.gamma, cp.arange(seq_len, dtype=cp.float32))
        discount_matrix = cp.tril(cp.outer(discount_factors, cp.ones(seq_len)))
        returns = discount_matrix @ rewards
        
        x, y, vx, vy = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
        self.returns[x, y, vx, vy, actions] += returns
        self.counts[x, y, vx, vy, actions] += 1
        
        valid_counts = cp.clip(self.counts, 1, None)  # Prevent division by zero
        self.q_function[x, y, vx, vy, actions] += (returns - self.q_function[x, y, vx, vy, actions]) / valid_counts[x, y, vx, vy, actions]
        self.policy[x, y, vx, vy] = cp.argmax(self.q_function[x, y, vx, vy])    
    
    def choose_action(self, state):
        """
        Selects an action using an epsilon-greedy strategy.
        """
        if cp.random.rand() < self.epsilon:
            action_index = cp.random.randint(0, self.num_actions)
        else:
            action_index = self.policy[tuple(state)]
        
        self.epsilon = max(self.mini_epsilon, self.epsilon * self.decay)
        return action_index
