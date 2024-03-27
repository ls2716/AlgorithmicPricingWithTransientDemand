"""Utils module that defines utility functions for the project"""

import logging
import os
import numpy as np
import matplotlib.pyplot as plt


def get_logger(name, level=logging.INFO):
    """Return a logger object"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = get_logger(__name__)


def create_folder(folder):
    """Create a folder if it does not exist"""
    if not os.path.exists(folder):
        os.makedirs(folder)


def single_game(env, bandits, T, noise=None):
    """Run a single game"""
    # Reset the bandits
    for bandit in bandits:
        bandit.reset()
    env.reset()
    # Initialise the action history
    action_history = np.zeros((len(bandits), T))
    # Initialise the rewards history
    rewards_history = np.zeros((len(bandits), T))
    # Initialise the randomised history
    randomised_history = np.zeros((len(bandits), T))
    # Run the game
    for t in range(T):
        # Get the actions
        action_tuple = [bandit.get_action() for bandit in bandits]
        actions = np.array([action for action, _, _ in action_tuple])
        action_indices = np.array(
            [action_index for _, action_index, _ in action_tuple])
        action_randomised = np.array(
            [action_randomised for _, _, action_randomised in action_tuple])
        # Get the rewards
        _, rewards = env.get_reward(actions.reshape(1, -1))
        rewards = rewards.reshape(-1)
        # Add noise if needed
        if noise is not None:
            rewards += noise * np.random.randn(len(bandits))

        for i, bandit in enumerate(bandits):
            action_index = action_indices[i]
            bandit.update(action_index, rewards[i])
        rewards_history[:, t] = rewards
        action_history[:, t] = actions
        randomised_history[:, t] = action_randomised
    return rewards_history, action_history, randomised_history


def plot_reward_history(reward_history, bandit_names, folder, filename, show_plot=False):
    plt.figure()
    if bandit_names is None:
        bandit_names = ['Bandit '+str(i)
                        for i in range(reward_history.shape[0])]
    for i, rewards in enumerate(reward_history):
        plt.plot(reward_history[i, :], label=bandit_names[i])
    plt.xlabel('Time step')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(folder, filename))
    if show_plot:
        plt.show()
    plt.close()


def plot_action_history(action_history, bandit_names, folder, filename, show_plot=False):
    plt.figure()
    if bandit_names is None:
        bandit_names = ['Bandit '+str(i)
                        for i in range(action_history.shape[0])]
    for i, actions in enumerate(action_history):
        plt.plot(action_history[i, :], label=bandit_names[i])
    plt.xlabel('Time step')
    plt.ylabel('Action')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(folder, filename))
    if show_plot:
        plt.show()
    plt.close()


def plot_reward_and_action(reward_history, action_history, bandit_names, folder, filename, show_plot=False,
                           nash_payoffs=None, pareto_payoffs=None):
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    if bandit_names is None:
        bandit_names = ['Bandit '+str(i+1)
                        for i in range(reward_history.shape[0])]
    for i, rewards in enumerate(reward_history):
        axs[0].plot(reward_history[i, :], label=bandit_names[i])
    for i, actions in enumerate(action_history):
        axs[1].plot(action_history[i, :], label=bandit_names[i])
    axs[0].set_xlabel('Time step')
    axs[0].set_ylabel('Reward')
    if nash_payoffs is not None and pareto_payoffs is not None:
        secondary_axis = axs[0].twinx()
        ylims = axs[0].get_ylim()
        new_ylims = normalise_payoffs(np.array(ylims).reshape(-1, 1),
                                      nash_payoffs, pareto_payoffs)
        secondary_axis.set_ylabel('Normalised reward $\Delta$')
        secondary_axis.set_ylim(new_ylims)
    axs[0].legend()
    axs[1].set_xlabel('Time step')
    axs[1].set_ylabel('Price')
    # Add second axis for margins
    secondary_axis = axs[1].twinx()
    ylims = np.array(axs[1].get_ylim()).reshape(-1, 1)
    new_ylims = ylims - 1.
    secondary_axis.set_ylabel('Margin')
    secondary_axis.set_ylim(new_ylims)
    axs[1].legend()
    axs[0].grid()
    axs[1].grid()
    plt.savefig(os.path.join(folder, filename))
    if show_plot:
        plt.show()
    plt.close()


def find_nash(env, dp=0.001):
    N = env.N

    pm = np.array(env.Cs[0])

    # Define starting prices
    prices = np.ones(N)*pm
    last_reward = -np.inf
    demands, rewards = env.get_reward(prices)
    current_reward = rewards[0, -1]
    while current_reward > last_reward:
        prices[:] = prices[-1]
        demands, rewards = env.get_reward(prices)
        last_reward = rewards[0, -1]
        prices[-1] += dp
        demands, rewards = env.get_reward(prices)
        current_reward = rewards[0, -1]

    prices[:] = prices[0]
    logger.info(f'Nash prices: {prices}')
    logger.info(f'Nash reward: {last_reward}')
    return prices, last_reward


def find_pareto(env, dp=0.001):
    N = env.N

    pm = np.array(env.Cs[0])

    # Define starting prices
    prices = np.ones(N)*pm
    last_reward = -np.inf
    demands, rewards = env.get_reward(prices)
    current_reward = rewards[0, -1]
    while current_reward > last_reward:
        prices[:] = prices[-1]
        demands, rewards = env.get_reward(prices)
        last_reward = rewards[0, -1]
        prices[:] += dp
        demands, rewards = env.get_reward(prices)
        current_reward = rewards[0, -1]

    prices[:] = prices[0]
    logger.info(f'Pareto prices: {prices}')
    logger.info(f'Pareto reward: {last_reward}')
    return prices, last_reward


def normalise_payoffs(rewards, nash_payoffs, pareto_payoffs):
    norm_rewards = (rewards - nash_payoffs.reshape(-1, 1)) / \
        (pareto_payoffs.reshape(-1, 1)-nash_payoffs.reshape(-1, 1))
    return norm_rewards


def plot_reward_and_action_randomised(reward_history, action_history, randomised_history,
                                      bandit_names, folder, filename, show_plot=False,
                                      nash_payoffs=None, pareto_payoffs=None):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    if bandit_names is None:
        bandit_names = ['Bandit '+str(i+1)
                        for i in range(reward_history.shape[0])]
    for i, rewards in enumerate(reward_history):
        axs[0].plot(reward_history[i, :], label=bandit_names[i])
    for i, actions in enumerate(action_history):
        axs[1].plot(action_history[i, :], label=bandit_names[i])
    for i, randomised in enumerate(randomised_history):
        axs[2].plot(randomised_history[i, :], label=bandit_names[i])
    axs[0].set_xlabel('Time step')
    axs[0].set_ylabel('Reward')
    if nash_payoffs is not None and pareto_payoffs is not None:
        secondary_axis = axs[0].twinx()
        ylims = axs[0].get_ylim()
        new_ylims = normalise_payoffs(np.array(ylims).reshape(-1, 1),
                                      nash_payoffs, pareto_payoffs)
        secondary_axis.set_ylabel('normalised reward $\Delta$')
        secondary_axis.set_ylim(new_ylims)
    axs[0].legend()
    axs[0].grid()
    axs[1].set_xlabel('Time step')
    axs[1].set_ylabel('Price')
    # Add second axis for margins
    secondary_axis = axs[1].twinx()
    ylims = np.array(axs[1].get_ylim()).reshape(-1, 1)
    new_ylims = ylims - 1.
    secondary_axis.set_ylabel('margin')
    secondary_axis.set_ylim(new_ylims)
    axs[1].legend()
    axs[1].grid()
    axs[2].set_xlabel('Time step')
    axs[2].set_ylabel('Randomised')
    axs[2].legend()
    axs[0].grid('on', which='both')
    axs[1].grid('on', which='both')
    axs[2].grid('on', which='both')

    plt.savefig(os.path.join(folder, filename))
    if show_plot:
        plt.show()
    plt.close()
