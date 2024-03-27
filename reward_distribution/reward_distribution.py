"""This script computes the reward distribution for exploring agent.
"""
# Import numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Import the environment
from environment import Environment


# Import logging utility and define logger
import utils as ut
logger = ut.get_logger(__name__)


# Define the half-width
half_width = 0.005
# Define the resolution
dp = 0.001
# Define the epsilon
eps = 0.25


# Define the number of samples
no_samples = 100000

# Define the number of agents
N = 2

# Define parameters for the environment
As = np.array([1.]*N)
Cs = np.array([1.]*N)
a_0 = -1.
mu = 0.25

env = Environment(N, As, Cs, mu, a_0)

# Compute Nash equilibrium using ut.find_nash
nash_prices, nash_reward = ut.find_nash(env)


def reward_distribution(env, current_price, half_width, epsilon, N):
    # Define price range
    price_array = np.linspace(current_price-half_width, current_price+half_width,
                              int(half_width*2/dp+1), endpoint=True)
    logger.info(f'Prices: {price_array}')
    reward_array = []
    # Calculate base reward
    prices = np.ones((no_samples, N))*current_price
    prices[:, 0] = current_price
    # Generate randomised price vectors
    for i in range(1, N):
        randomised_prices = np.random.choice(price_array, no_samples)
        explore = np.random.rand(no_samples) < epsilon
        prices[explore, i] = randomised_prices[explore]
    demands, rewards = env.get_reward(prices)
    base_reward = np.mean(rewards[:, 0])
    prob_array = []
    for price in price_array:
        # Generate price vectors
        prices = np.ones((no_samples, N))*current_price
        prices[:, 0] = price
        # Generate randomised price vectors
        for i in range(1, N):
            randomised_prices = np.random.choice(price_array, no_samples)
            explore = np.random.rand(no_samples) < epsilon
            prices[explore, i] = randomised_prices[explore]
        logger.info(f'Prices: {prices[:10,:]}')
        # Get rewards
        demands, rewards = env.get_reward(prices)
        # logger.info(f'Rewards: {rewards}')
        # Get mean reward for the first agent
        mean_reward = np.mean(rewards[:, 0])
        logger.info(f'Price: {price}')
        logger.info(f'Mean reward: {mean_reward}')
        reward_array.append(mean_reward)

        # Calculate the probability that reward is greater than the base reward
        prob = np.mean(rewards[:, 0] > base_reward)
        prob_array.append(prob)
        logger.info(f'Probability: {prob}')
        # Plot the distribution of rewards
        # logger.info(f'Reward distribution for price {price}:')
        # plt.hist(rewards[:, 0], bins=100)
        # plt.xlabel('Reward')
        # plt.ylabel('Frequency')
        # plt.grid()
        # plt.show()
        # plt.close()
    # Plot the rewards
    plt.plot(price_array, prob_array)
    plt.xlabel('Price')
    plt.ylabel('Mean reward')
    plt.grid()
    plt.show()
    plt.close()


reward_distribution(env, 1.48, half_width, eps, N)
