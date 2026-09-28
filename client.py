"""Soft Actor-Critic (SAC) Maximum Entropy Engine.
100% Python Standard Library.
"""

import math

class SACMaximumEntropyEvaluator:
    """Soft Actor-Critic maximum entropy policy value and soft Bellman backup."""
    def __init__(self, alpha=0.2, gamma=0.99):
        self.alpha = alpha
        self.gamma = gamma

    def soft_value(self, q_values, policy_probs):
        v = 0.0
        entropy = 0.0
        for q, p in zip(q_values, policy_probs):
            if p > 1e-12:
                log_p = math.log(p)
                entropy -= p * log_p
                v += p * (q - self.alpha * log_p)
        return {"soft_value": v, "entropy": entropy}

    def soft_bellman_target(self, reward, next_soft_value, done=False):
        return reward if done else reward + self.gamma * next_soft_value
