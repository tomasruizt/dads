from functools import partial

import pytest
import numpy as np

from envs.fetch import make_fetch_pick_and_place_env, make_fetch_slide_env, _get_goal_from_state_fetch, \
    FixedGoalFetchPickAndPlaceEnv, FixedGoalFetchSlideEnv, make_point2d_dads_env

envs_fns = [
    make_fetch_pick_and_place_env,
    make_fetch_slide_env,
    make_point2d_dads_env
]


@pytest.fixture(params=envs_fns)
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

    # Vectorized
    many_obs = np.asarray([env.observation_space.sample() for _ in range(99)])
    assert np.allclose(0, potential(many_obs, many_obs))
    assert len(many_obs) == len(potential(many_obs, many_obs))

    other_obs = np.asarray([env.observation_space.sample() for _ in range(99)])
    vec_potential = potential(many_obs, other_obs)
    assert np.allclose(vec_potential, -potential(other_obs, many_obs))
    assert not np.allclose(0, vec_potential)
    assert len(other_obs) == len(vec_potential)


@pytest.mark.parametrize("env_ctor", [FixedGoalFetchPickAndPlaceEnv, FixedGoalFetchSlideEnv])
def test_block_pos(env_ctor):
    env = env_ctor()
    obs = env.reset()
    assert np.allclose(obs["achieved_goal"], _get_goal_from_state_fetch(obs["observation"]))
