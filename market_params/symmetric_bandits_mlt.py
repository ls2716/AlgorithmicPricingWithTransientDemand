"""Scipt which runs the symmetric 3 bandits problem."""

import numpy as np
import os

from bandits import LocalBandit
from environment import Environment, DelayedDemandEnv
import utils as ut


logger = ut.get_logger(__name__)

# Create folder name
image_folder = '_images/market_params'
results_folder = '_results/market_params'
root_data_folder = '_data/market_params'
ut.create_folder(image_folder)
ut.create_folder(results_folder)
ut.create_folder(root_data_folder)

# Define the environment
N = 3
mu = 1
As = [1.] * N
Cs = [1.] * N
a_0 = 1
env = Environment(N, As, Cs, mu, a_0)
delay = 2
delayed_env = DelayedDemandEnv(env, delay)

# Find the Nash prices
nash_prices, nash_payoffs = ut.find_nash(env, 0.0001)
# Find the Pareto prices
pareto_prices, pareto_payoffs = ut.find_pareto(env, 0.0001)

logger.info(f"Nash prices: {nash_prices}")
logger.info(f"Nash payoffs: {nash_payoffs}")

rewards = env.get_reward(nash_prices)
logger.info(f"Nash rewards: {rewards}")

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

# Define number of simulations
no_sim = 10
# Run the game
T = 20000
initial_T = 5000
all_rewards = np.zeros((no_sim, N, T))
all_actions = np.zeros((no_sim, N, T))
for sim in range(no_sim):
    logger.info(f'Simulation {sim+1}')
    rewards, actions, randomised = ut.single_game(delayed_env, bandits, T)
    mean_rewards = np.mean(rewards[:, initial_T:], axis=1)
    mean_actions = np.mean(actions[:, initial_T:], axis=1)
    logger.info(
        f"Mean rewards excluding first {initial_T} steps: %s", mean_rewards)
    logger.info(
        f"Mean actions excluding first {initial_T} steps: %s", mean_actions)
    all_rewards[sim] = rewards
    all_actions[sim] = actions

norm_rewards = ut.normalise_payoffs(rewards, nash_payoffs, pareto_payoffs)
mean_norm_rewards = np.mean(norm_rewards[:, initial_T:], axis=1)
logger.info(
    f"Mean normalized rewards excluding first {initial_T} steps: %s", mean_rewards)

# Plot the last rewards
ut.plot_reward_and_action(rewards, actions, None,
                          image_folder, f'bandits_{delay}_delayed_{T}_mu_{mu:.2f}_a0_{a_0:.2f}_mlt_{no_sim}.png',
                          nash_payoffs=nash_payoffs, pareto_payoffs=pareto_payoffs,
                          nash_price=nash_prices[0], cost=Cs[0])


# Save arrays to a file
data_folder = os.path.join(
    root_data_folder, f"data_bandits_{delay}_delayed_{T}_mu_{mu:.2f}_a0_{a_0:.2f}_mlt_{no_sim}")
ut.create_folder(data_folder)
np.save(os.path.join(data_folder,
        f"bandits_{delay}_delayed_{T}_rewards.npy"), all_rewards)
np.save(os.path.join(data_folder,
        f"bandits_{delay}_delayed_{T}_actions.npy"), all_actions)
# Save other information as a json
ut.save_json(data_folder,  f"bandits_{delay}_delayed_{T}.json", {
    'N': N,
    'As': As,
    'Cs': Cs,
    'mu': mu,
    'a_0': a_0,
    'delay': delay,
    'epsilon': epsilon,
    'tau': tau,
    'half_width': half_width,
    'T': T,
    'initial_T': initial_T,
    'no_sim': no_sim,
    'nash_prices': nash_prices.tolist(),
    'nash_payoffs': nash_payoffs,
    'pareto_prices': pareto_prices.tolist(),
    'pareto_payoffs': pareto_payoffs
})

# Save results
ut.reduce_results(all_rewards, all_actions, initial_T,
                  nash_payoffs, nash_prices[0], pareto_payoffs, Cs[0],
                  results_folder, f"bandits_{delay}_delayed_{T}_mu_{mu:.2f}_a0_{a_0:.2f}_mlt_{no_sim}.txt")
