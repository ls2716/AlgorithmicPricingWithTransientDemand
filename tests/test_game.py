"""Test the single game function"""
import numpy as np
import pytest
from bandits import LocalBandit
from environment import Environment, DelayedDemandEnv
from utils import single_game, get_logger


logger = get_logger(__name__)


def test_single_game():
    """Test the single game function"""
    # Set the random seed
    np.random.seed(0)
    # Define the environment
    env = Environment(2, [1, 1], [1, 1], 0.25, 0.)
    delayed_env = DelayedDemandEnv(env, 1)
    # Define the bandits
    bandits = [LocalBandit(0.3, np.linspace(1, 4, 301, endpoint=True), 30, 5)
               for _ in range(2)]
    # Run the game
    rewards, actions = single_game(delayed_env, bandits, 200)
    print(rewards, actions)
    mean_rewards = np.mean(rewards, axis=1)
    logger.info("Mean rewards: %s", mean_rewards)
    assert np.array([0.00023839, 0.06455505]) == pytest.approx(
        mean_rewards, abs=1e-6)
    return None
