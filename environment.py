"""The script implements the environment for bandits"""
"""This file specifies the environment for the game and 
the corresponding functions which return the rewards."""

# Define a logger
import numpy as np
import logging
import utils as ut
logger = ut.get_logger(__name__)


class Environment():
    """The pricing environment for the game."""

    def __init__(self, N, As, Cs, mu, a_0):
        """
        Arguments:
            - N (int): number of agents
            - As (list of floats): list of product quality indexes
            - mu (float): index of horizontal differentitation
            - a_0 (float) inverse index of aggregate demand
            - Cs (list of floats): list of agent costs per product
        """
        self.N = N
        self.As = np.array(As).reshape(-1)
        self.Cs = np.array(Cs).reshape(-1)
        self.mu = mu
        self.a_0 = a_0

    def reset(self):
        """Reset the environment."""
        pass

    def get_reward(self, prices):
        """Step function.

        Given array of prices returns array of rewards.

        Arguments:
            - P (numpy array of floats): list of agent prices
        """
        # Fix prices shape
        prices = prices.reshape(-1, self.N)
        # Compute demands
        demands = np.exp((self.As-prices)/self.mu)
        demands = demands / \
            (np.sum(np.exp((self.As-prices)/self.mu), axis=1)
             + np.exp(self.a_0/self.mu)).reshape(-1, 1)
        # Return rewards
        return demands, demands*(prices-self.Cs)


class DelayedDemandEnv():
    """Pricing environment with delayed demand."""

    def __init__(self, environment, delay):
        """
        Arguments:
            - environment (Environment): the environment
            - delay (int): the delay
        """
        self.environment = environment
        self.delay = delay
        self.demands_history = None
        self.N = environment.N

    def reset(self):
        """Reset the environment"""
        self.demands_history = None

    def get_reward(self, prices):
        demands, rewards = self.environment.get_reward(prices)
        # If there is no delay, return the rewards
        if self.delay == 1:
            return demands, rewards
        # If there is a delay, store the demands in the history
        if self.demands_history is None:
            # If the history is empty, initialise it
            self.demands_history = np.copy(demands).reshape(-1, self.N, 1)
        else:
            # else append current demands to the history and remove the oldest demands
            self.demands_history = np.concatenate(
                (self.demands_history[:, :, -(self.delay-1):], demands.reshape(-1, self.N, 1)), axis=2)
        # Return the average demands and rewards
        delayed_demands = np.mean(self.demands_history, axis=2)
        return delayed_demands, delayed_demands*(prices-self.environment.Cs)


if __name__ == "__main__":
    # Define the environment parameters
    N = 3
    As = [1.] * N
    Cs = [1.] * N
    a_0 = -1
    mu = 1.
    # Create the environment
    env = Environment(N, As, Cs, mu, a_0)
    # Define the prices
    prices = np.array([[1., 1., 1.], [2., 2.1, 2.]])
    # Get the rewards
    delayed = DelayedDemandEnv(env, 2)

    delayed.get_reward(prices)
    prices = np.array([[1., 1.5, 1.5], [2.5, 2.1, 2.]])
    delayed.get_reward(prices)
    delayed.get_reward(prices)
