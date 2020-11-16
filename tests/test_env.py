from functools import partial
import pytest
import numpy as np

from envs.custom_envs import make_fetch_pick_and_place_env, make_fetch_slide_env, \
    CustomFetchPickAndPlaceEnv, FixedGoalFetchSlideEnv, make_point2d_dads_env, \
    DADSCustomFetchReachEnv, make_fetch_reach_env, make_toylab_dads_env, DADSEnv

convex_envs_fns = [
    make_fetch_reach_env,
    make_fetch_pick_and_place_env,
    make_fetch_slide_env,
    make_point2d_dads_env,
]

envs_fns = [
    *convex_envs_fns,
    make_toylab_dads_env
]

fetch_env_ctors = [
    CustomFetchPickAndPlaceEnv,
    FixedGoalFetchSlideEnv,
    DADSCustomFetchReachEnv
]

VECTORIZED_DIM = 50  # dim to test vectorized funcs


@pytest.fixture(params=envs_fns)
def env_fn(request):
    yield request.param


def test_env_trajectories_dont_env(env_fn):
    env = env_fn()
    for _ in range(3):
        env.reset()
        for _ in range(10):
            env.step(env.action_space.sample())

    dones = [env.step(env.action_space.sample())[2] for _ in range(150)]
    assert len(dones) > 0 and not any(dones)


@pytest.mark.parametrize("env_fn", convex_envs_fns)
def test_reward_is_convex(env_fn):
    env: DADSEnv = env_fn()
    obs_type = DADSEnv.OBS_TYPE.DYNAMICS_OBS
    obs = np.asarray([_rand_obs(env, obs_type=obs_type) for _ in range(5)])
    goal = np.vstack([env.goal] * 5)
    reward = partial(env.compute_reward, info=None)
    assert np.all(reward(obs, goal) < reward(_closer(obs, goal), goal))


def _closer(x_from, x_to):
    return 0.5 * (x_from + x_to)


@pytest.mark.parametrize("obs_type",
                         [DADSEnv.OBS_TYPE.FULL_OBS, DADSEnv.OBS_TYPE.DYNAMICS_OBS])
def test_env_dads_reward_fn(env_fn, obs_type):
    env: DADSEnv = env_fn()
    env.observation_space.seed(0)
    env.reset()
    new_rand_obs = lambda: _rand_obs(env=env, obs_type=obs_type)
    obs, random_obs = new_rand_obs(), new_rand_obs()

    info, max_reward = None, 0
    assert np.isclose(max_reward, env.compute_reward(obs, obs, info=info))
    assert env.compute_reward(obs, random_obs, info=info) < max_reward

    potential = partial(env.compute_reward, info=obs_type)
    assert np.isclose(max_reward, potential(random_obs, random_obs))

    winning_obs = _new_winning_obs(env, obs_type=obs_type)
    assert not np.isclose(max_reward, potential(winning_obs, obs))
    assert np.isclose(potential(winning_obs, obs), -potential(obs, winning_obs))

    # Vectorized
    many_obs = np.asarray([new_rand_obs() for _ in range(VECTORIZED_DIM)])
    assert np.allclose(0, potential(many_obs, many_obs))
    assert len(many_obs) == len(potential(many_obs, many_obs))

    other_obs = np.asarray([new_rand_obs() for _ in range(VECTORIZED_DIM)])
    vec_potential = potential(many_obs, other_obs)
    assert np.allclose(vec_potential, -potential(other_obs, many_obs))
    assert not np.allclose(0, vec_potential)
    assert len(other_obs) == len(vec_potential)


def _rand_obs(env: DADSEnv, obs_type: DADSEnv.OBS_TYPE):
    obs = env.observation_space.sample()
    if obs_type == DADSEnv.OBS_TYPE.DYNAMICS_OBS:
        return env.to_dynamics_obs(obs)
    return obs


def _new_winning_obs(env, obs_type: DADSEnv.OBS_TYPE):
    obs = env.observation_space.sample()
    achieved_goal = env.achieved_goal_from_state(obs)
    obs[[e in achieved_goal for e in obs]] = env.goal.copy()
    if obs_type == DADSEnv.OBS_TYPE.DYNAMICS_OBS:
        return env.to_dynamics_obs(obs)
    return obs


def test_to_dynamics_obs_fn(env_fn):
    env: DADSEnv = env_fn()
    obs = env.observation_space.sample()
    dynamics_obs = env.to_dynamics_obs(obs)
    assert np.allclose(dynamics_obs, env.achieved_goal_from_state(obs))
    assert len(dynamics_obs) == env.dyn_obs_dim()


@pytest.mark.parametrize("env_ctor", fetch_env_ctors)
def test_goal_pos(env_ctor):
    env = env_ctor()
    obs = env.reset()
    assert np.allclose(obs["achieved_goal"], env.achieved_goal_from_state(obs["observation"]))

    # Vectorized
    many_obs = np.repeat(obs["observation"][None], VECTORIZED_DIM, axis=0)
    achieved_goals = np.repeat(obs["achieved_goal"][None], VECTORIZED_DIM, axis=0)
    assert np.allclose(achieved_goals, env.achieved_goal_from_state(many_obs))
    assert len(env.achieved_goal_from_state(many_obs)) == VECTORIZED_DIM
