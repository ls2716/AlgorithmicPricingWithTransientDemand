"""This script contains the bandit classes for the game."""

import numpy as np
import logging
import utils as ut
from collections import deque

# Define a logger
logger = ut.get_logger(__name__)


class LocalBandit():
    """The non-stationary epsilon-greedy bandit
    which explores around current best action."""

    def __init__(self, eps, action_set, tau: int, half_width: int) -> None:
        """
        Arguments:
            - eps (float): probability of exploration
            - half_width (float): half_width of exploration
            - tau (int): the sliding window size
        """
        self.eps = eps
        self.half_width = half_width
        self.m = np.zeros(len(action_set))
        self.tau = tau
        self.current_action = np.random.randint(0, len(action_set))
        self.action_set = action_set
        self.rewards = None
        self.zero_step = True
        self.starting_action = None

    def reset(self):
        """Reset the bandit."""
        self.m = np.zeros(len(self.m))
        self.rewards = None
        self.current_action = np.random.randint(0, len(self.action_set))
        self.rewards = None
        self.zero_step = True
        if self.starting_action is not None:
            self.m[self.starting_action] = 1.

    def set_starting_action(self, action):
        """Set the starting action."""
        action_index = np.argmin(np.abs(self.action_set - action))
        self.starting_action = action_index

    def get_action(self):
        """Return the action."""
        # Find the maximum action value
        max_value = np.max(self.m)
        # Get actions with maximum value
        max_indices = np.where(self.m == max_value)[0]
        # Sample a maximising action
        max_index = np.random.choice(max_indices)
        # Epsilon-greedy action selection
        if np.random.rand() < self.eps and not self.zero_step:
            action_randomised = True
            rand_add = np.random.randint(-self.half_width, self.half_width+1)
            action_index = np.maximum(np.minimum(
                len(self.action_set)-1, max_index + rand_add), 0)
        else:
            action_index = max_index
            action_randomised = False
        if self.zero_step:
            self.zero_step = False
        # Return action and action index
        return self.action_set[action_index], action_index, self.action_set[max_index]

    def update_old(self, action_index, reward):
        """Update the action-value function."""
        # Update the rewards
        if self.rewards is None:
            # If the rewards are not initialised, initialise them
            self.rewards = np.zeros(len(self.action_set)).reshape(1, -1)
            self.rewards[0, action_index] = reward
        else:
            # Othewise append new rewards to the history and remove the oldest rewards
            step_rewards = np.zeros(len(self.action_set)).reshape(1, -1)
            step_rewards[0, action_index] = reward
            self.rewards = np.vstack(
                (self.rewards[(-self.tau+1):], step_rewards))
        # Update the action-value function
        self.m = np.sum(self.rewards, axis=0) / \
            np.maximum(np.sum(self.rewards != 0, axis=0), 1)

    def update(self, action_index, reward):
        """Update the action-value function."""
        # Update the rewards
        if self.rewards is None:
            # If the rewards are not initialised, initialise them
            self.rewards = deque()
            self.indices = deque()
            self.sums = np.zeros(len(self.action_set))
            self.counters = np.zeros(len(self.action_set))
        step_rewards = np.zeros(len(self.action_set)).reshape(-1)
        step_rewards[action_index] = reward
        self.rewards.append(step_rewards)
        self.indices.append(action_index)
        self.sums[action_index] += reward
        self.counters[action_index] += 1
        self.m[action_index] = self.sums[action_index] / \
            self.counters[action_index]
        # Remove the oldest reward
        if len(self.rewards) > self.tau:
            oldest_rewards = self.rewards.popleft()
            oldest_index = self.indices.popleft()
            self.sums[oldest_index] -= oldest_rewards[oldest_index]
            self.counters[oldest_index] -= 1
            if self.counters[oldest_index] == 0:
                self.m[oldest_index] = 0.
            else:
                self.m[oldest_index] = self.sums[oldest_index] / \
                    self.counters[oldest_index]


if __name__ == "__main__":

    # Initialise a local bandit
    bandit = LocalBandit(0.1, [0, 0.1, 0.2, 0.3, 0.4, 0.5], 3, 1)
    # Set seed
    np.random.seed(0)
    for i in range(5):
        # Get the action
        action, action_index, _ = bandit.get_action()
        # Update the bandit
        bandit.update(action_index, -i-1)
        print(bandit.m)
        print(bandit.rewards)
