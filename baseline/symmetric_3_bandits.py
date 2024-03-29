"""Scipt which runs the symmetric 3 bandits problem."""

import numpy as np
import os

from bandits import LocalBandit
from environment import Environment, DelayedDemandEnv
import utils as ut


logger = ut.get_logger(__name__)

# Create folder name
image_folder = '_images/baseline'
results_folder = '_results/baseline'
ut.create_folder(image_folder)
ut.create_folder(results_folder)

# Define the environment
N = 3
mu = 0.25
As = [1.] * N
Cs = [1.] * N
a_0 = -1.
env = Environment(N, As, Cs, mu, a_0)
delay = 1
delayed_env = DelayedDemandEnv(env, delay)

# Find the Nash prices
nash_prices, nash_payoffs = ut.find_nash(env, 0.0001)
# Find the Pareto prices
pareto_prices, pareto_payoffs = ut.find_pareto(env, 0.0001)


# Define the bandits
action_set = np.linspace(1, 4, 3001, endpoint=True)
half_width = 5
epsilon = 0.25
tau = 50
bandits = [LocalBandit(epsilon, action_set, tau, half_width)
           for _ in range(N)]

# Set starting action
starting_action = nash_prices[0]
for bandit in bandits:
    bandit.set_starting_action(starting_action)

logger.info(f'Half width: {action_set[0]} - {action_set[half_width]}')

# Set the random seed
np.random.seed(0)

# Run the game
T = 2000
initial_T = 1000
rewards, actions, randomised = ut.single_game(delayed_env, bandits, T)
mean_rewards = np.mean(rewards[:, initial_T:], axis=1)
mean_actions = np.mean(actions[:, initial_T:], axis=1)
logger.info(
    f"Mean rewards excluding first {initial_T} steps: %s", mean_rewards)
logger.info(
    f"Mean actions excluding first {initial_T} steps: %s", mean_actions)

norm_rewards = ut.normalise_payoffs(rewards, nash_payoffs, pareto_payoffs)
mean_norm_rewards = np.mean(norm_rewards[:, initial_T:], axis=1)
logger.info(
    f"Mean normalized rewards excluding first {initial_T} steps: %s", mean_rewards)

# Plot the rewards
ut.plot_reward_and_action(rewards, actions, None,
                          image_folder, f'bandits_{delay}_delayed_{T}.png',
                          nash_payoffs=nash_payoffs, pareto_payoffs=pareto_payoffs,
                          nash_price=nash_prices[0], cost=Cs[0],
                          show_plot=False)


def round_4(x):
    return np.round(x, 4)


# Save results
with open(os.path.join(results_folder, f'bandits_{delay}_delayed_{T}.txt'), 'w+') as f:
    f.write(f'{round_4(nash_payoffs)} Nash payoffs \n')
    f.write(f'{round_4(pareto_payoffs)} Pareto payoffs \n')
    f.write('------------------------------------\n')
    f.write(
        f'{round_4(mean_rewards)} Mean rewards excluding first {initial_T} steps \n')
    f.write(
        f'{round_4((mean_rewards)/(nash_payoffs))*100} % of Nash payoffs\n')
    f.write(
        f'{round_4(mean_norm_rewards)} Normalised mean rewards excluding first {initial_T} steps\n')
    f.write('------------------------------------\n')
    f.write(f'{round_4(nash_prices)} Nash prices\n')
    f.write(
        f'{round_4(mean_actions)} Mean actions excluding first {initial_T} steps\n')
    f.write('------------------------------------\n')
    f.write(f'{round_4(nash_prices - Cs)} Nash margins\n')
    f.write(
        f'{round_4(mean_actions - Cs)} Mean margins excluding first {initial_T} steps\n')
    f.write(
        f'{round_4((mean_actions - Cs)/(nash_prices - Cs))*100} % of Nash margins\n')


# # Plot the rewards from small range
# ut.plot_reward_and_action_randomised(rewards, actions, randomised, None,
#                                      image_folder, f'bandits_{delay}_delayed_{T}_.png',
#                                      nash_payoffs=nash_payoffs, pareto_payoffs=pareto_payoffs, show_plot=True)
