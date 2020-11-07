from functools import partial
import pytest
import numpy as np

from envs.fetch import make_fetch_pick_and_place_env, make_fetch_slide_env, \
    FixedGoalFetchPickAndPlaceEnv, FixedGoalFetchSlideEnv, make_point2d_dads_env, \
    FixedGoalFetchReach, make_fetch_reach_env

envs_fns = [
    make_fetch_pick_and_place_env,
    make_fetch_slide_env,
    make_fetch_reach_env,
    make_point2d_dads_env
]

fetch_env_ctors = [
    FixedGoalFetchPickAndPlaceEnv,
    FixedGoalFetchSlideEnv,
    FixedGoalFetchReach
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


@pytest.mark.parametrize("env_ctor", fetch_env_ctors)
def test_block_pos(env_ctor):
    env = env_ctor()
    obs = env.reset()
    assert np.allclose(obs["achieved_goal"], env.achieved_goal_from_state(obs["observation"]))

    # Vectorized
    dims = 7
    many_obs = np.repeat(obs["observation"][None], dims, axis=0)
    achieved_goals = np.repeat(obs["achieved_goal"][None], dims, axis=0)
    assert np.allclose(achieved_goals, env.achieved_goal_from_state(many_obs))
    assert len(env.achieved_goal_from_state(many_obs)) == dims
