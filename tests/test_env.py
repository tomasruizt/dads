from functools import partial
import pytest
import numpy as np

from envs.custom_envs import make_fetch_pick_and_place_env, make_fetch_slide_env, \
    FixedGoalFetchPickAndPlaceEnv, FixedGoalFetchSlideEnv, make_point2d_dads_env, \
    DADSCustomFetchReachEnv, make_fetch_reach_env, make_toylab_dads_env

envs_fns = [
    make_fetch_pick_and_place_env,
    make_fetch_slide_env,
    make_fetch_reach_env,
    make_point2d_dads_env,
    make_toylab_dads_env
]

fetch_env_ctors = [
    FixedGoalFetchPickAndPlaceEnv,
    FixedGoalFetchSlideEnv,
    DADSCustomFetchReachEnv
]

VECTORIZED_DIM = 50  # dim to test vectorized funcs


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
    env.seed(0)
    obs = env.reset()
    random_obs = env.observation_space.sample()

    info, max_reward = None, 0
    assert np.isclose(max_reward, env.compute_reward(obs, obs, info=info))
    assert env.compute_reward(obs, random_obs, info=info) < max_reward

    potential = partial(env.compute_reward, info="dads")
    assert np.isclose(max_reward, potential(random_obs, random_obs))

    winning_obs = make_winning_obs(env)
    assert not np.isclose(max_reward, potential(winning_obs, obs))
    assert np.isclose(potential(winning_obs, obs), -potential(obs, winning_obs))

    # Vectorized
    many_obs = np.asarray([env.observation_space.sample() for _ in range(VECTORIZED_DIM)])
    assert np.allclose(0, potential(many_obs, many_obs))
    assert len(many_obs) == len(potential(many_obs, many_obs))

    other_obs = np.asarray([env.observation_space.sample() for _ in range(VECTORIZED_DIM)])
    vec_potential = potential(many_obs, other_obs)
    assert np.allclose(vec_potential, -potential(other_obs, many_obs))
    assert not np.allclose(0, vec_potential)
    assert len(other_obs) == len(vec_potential)


def make_winning_obs(env):
    obs = env.observation_space.sample()
    achieved_goal = env.achieved_goal_from_state(obs)
    obs[[e in achieved_goal for e in obs]] = env.goal.copy()
    return obs


@pytest.mark.parametrize("env_ctor", fetch_env_ctors)
def test_block_pos(env_ctor):
    env = env_ctor()
    obs = env.reset()
    assert np.allclose(obs["achieved_goal"], env.achieved_goal_from_state(obs["observation"]))

    # Vectorized
    many_obs = np.repeat(obs["observation"][None], VECTORIZED_DIM, axis=0)
    achieved_goals = np.repeat(obs["achieved_goal"][None], VECTORIZED_DIM, axis=0)
    assert np.allclose(achieved_goals, env.achieved_goal_from_state(many_obs))
    assert len(env.achieved_goal_from_state(many_obs)) == VECTORIZED_DIM
