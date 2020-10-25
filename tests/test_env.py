from functools import partial

import pytest
import numpy as np

from envs.fetch import make_fetch_pick_and_place_env, make_fetch_slide_env, _block_pos, \
    FixedGoalFetchPickAndPlaceEnv, FixedGoalFetchSlideEnv


@pytest.fixture(params=[make_fetch_pick_and_place_env, make_fetch_slide_env])
def env_fn(request):
    yield request.param


def test_env_trajectory(env_fn):
    env = env_fn()
    for _ in range(3):
        env.reset()
        for _ in range(10):
            env.step(env.action_space.sample())


def test_env_reward_fn(env_fn):
    env = env_fn()
    obs = env.reset()
    random_obs = env.observation_space.sample()

    info, max_reward = None, 0
    assert np.isclose(max_reward, env.compute_reward(obs, obs, info=info))
    assert env.compute_reward(obs, random_obs, info=info) < max_reward

    potential = partial(env.compute_reward, info="dads")
    assert np.isclose(0, potential(random_obs, random_obs))
    assert not np.isclose(0, potential(random_obs, obs))
    assert np.isclose(potential(random_obs, obs), -potential(obs, random_obs))


@pytest.mark.parametrize("env_ctor", [FixedGoalFetchPickAndPlaceEnv, FixedGoalFetchSlideEnv])
def test_block_pos(env_ctor):
    env = env_ctor()
    obs = env.reset()
    assert np.allclose(obs["achieved_goal"], _block_pos(obs["observation"]))
