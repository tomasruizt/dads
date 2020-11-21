from abc import ABC
from enum import Enum

import sys
from collections import OrderedDict
from functools import partial

import numpy as np
from gym import Wrapper, ObservationWrapper, GoalEnv
from gym.envs.robotics import FetchPickAndPlaceEnv, FetchSlideEnv, FetchEnv, FetchReachEnv, \
    FetchPushEnv
from gym.wrappers import FilterObservation, FlattenObservation
from multi_goal.envs.toy_labyrinth_env import ToyLab

from envs.point2d_env import Point2DEnv


def make_toylab_dads_env(**kwargs):
    env = DADSCustomToyLabEnv()
    env = ObsAsOrderedDict(env)
    env = FilterObservation(env, filter_keys=["achieved_goal"])
    env = FlattenObservation(env)
    return DADSWrapper(env, **kwargs)


class DADSCustomToyLabEnv(ToyLab):
    def __init__(self):
        super().__init__(max_episode_len=sys.maxsize, use_random_starting_pos=True)

    @staticmethod
    def achieved_goal_from_state(state: np.ndarray) -> np.ndarray:
        return state[..., :2] if _is_batch(state) else state[:2]

    def compute_reward(self, achieved_obs, desired_obs, info):
        achieved_goal = self.achieved_goal_from_state(achieved_obs)
        desired_goal = self.achieved_goal_from_state(desired_obs)
        r = partial(super().compute_reward, info=info)
        if _is_batch(achieved_obs):
            return np.asarray([r(a, d) for a, d in zip(achieved_obs, desired_obs)])
        return r(achieved_goal=achieved_goal, desired_goal=desired_goal)

    @property
    def goal(self):
        return self._goal_pos


def _is_batch(x: np.ndarray) -> bool:
    assert x.ndim < 3
    return x.ndim == 2


class ObsAsOrderedDict(ObservationWrapper):
    def observation(self, observation):
        return OrderedDict(observation)


def make_point2d_dads_env(**kwargs):
    return DADSWrapper(Point2DEnv(), **kwargs)


def make_fetch_pick_and_place_env(**kwargs):
    return _process_fetch_env(CustomFetchPickAndPlaceEnv(reward_type="dense"), **kwargs)


def make_fetch_slide_env(**kwargs):
    return _process_fetch_env(FixedGoalFetchSlideEnv(reward_type="dense"), **kwargs)


def make_fetch_push_env(**kwargs):
    return _process_fetch_env(CustomFetchPushEnv(reward_type="dense"), **kwargs)


def make_fetch_reach_env(**kwargs):
    return _process_fetch_env(DADSCustomFetchReachEnv(reward_type="dense"), **kwargs)


def _process_fetch_env(env: FetchEnv, **kwargs):
    env = FilterObservation(env, filter_keys=["observation"])
    env = FlattenObservation(env)
    return DADSWrapper(env, **kwargs)


def _get_goal_from_state_fetch(state: np.ndarray) -> np.ndarray:
    return state[..., 3:6] if _is_batch(state) else state[3:6]


class CustomFetchPickAndPlaceEnv(FetchPickAndPlaceEnv):
    achieved_goal_from_state = staticmethod(_get_goal_from_state_fetch)


class CustomFetchPushEnv(FetchPushEnv):
    achieved_goal_from_state = staticmethod(_get_goal_from_state_fetch)


class FixedGoalFetchSlideEnv(FetchSlideEnv):
    _fixed_goal = np.asarray([1.7, 0.75, 0.41401894])

    achieved_goal_from_state = staticmethod(_get_goal_from_state_fetch)

    def _sample_goal(self):
        return self._fixed_goal.copy()


class DADSCustomFetchReachEnv(FetchReachEnv):
    @staticmethod
    def achieved_goal_from_state(state: np.ndarray) -> np.ndarray:
        return state[..., :3] if _is_batch(state) else state[:3]

    def _sample_goal(self):
        return np.random.uniform(low=[1.1, 0.4, 0.4], high=[1.45, 1.1, 0.8])


class DADSEnv(ABC, GoalEnv):
    goal: np.ndarray

    def dyn_obs_dim(self) -> int:
        raise NotImplementedError

    class OBS_TYPE(Enum):
        DYNAMICS_OBS = "DYNAMICS_OBS"
        FULL_OBS = "FULL_OBS"

    def to_dynamics_obs(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def achieved_goal_from_state(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DADSWrapper(Wrapper, DADSEnv):
    def __init__(self, env, use_state_space_reduction=True):
        super().__init__(env)
        self._use_state_space_reduction = use_state_space_reduction

    def compute_reward(self, achieved_goal, desired_goal, info):
        if info in DADSEnv.OBS_TYPE:
            return self._dads_reward(achieved_goal, desired_goal, obs_type=info)
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def _dads_reward(self, cur_obs, next_obs, obs_type: DADSEnv.OBS_TYPE):
        achieved_goal, next_achieved_goal = cur_obs, next_obs

        need_reduction = (obs_type == DADSEnv.OBS_TYPE.FULL_OBS
                          or obs_type == DADSEnv.OBS_TYPE.DYNAMICS_OBS and not self._use_state_space_reduction)
        if need_reduction:
            achieved_goal = self.env.achieved_goal_from_state(cur_obs)
            next_achieved_goal = self.env.achieved_goal_from_state(next_obs)

        goal = np.broadcast_to(self.goal, achieved_goal.shape)
        reward = lambda achieved: self.env.compute_reward(achieved, goal, info=None)
        return reward(next_achieved_goal) - reward(achieved_goal)

    def to_dynamics_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self._use_state_space_reduction:
            return obs
        return self.env.achieved_goal_from_state(state=obs)

    def achieved_goal_from_state(self, state: np.ndarray) -> np.ndarray:
        return self.env.achieved_goal_from_state(state=state)

    def dyn_obs_dim(self):
        return len(self.to_dynamics_obs(self.env.observation_space.sample()))
