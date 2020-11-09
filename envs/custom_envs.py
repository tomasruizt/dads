from abc import ABC
from enum import Enum

import sys
from collections import OrderedDict
from functools import partial

import numpy as np
from gym import Wrapper, ObservationWrapper, GoalEnv
from gym.envs.robotics import FetchPickAndPlaceEnv, FetchSlideEnv, FetchEnv, FetchReachEnv
from gym.wrappers import FilterObservation, FlattenObservation
from multi_goal.envs.toy_labyrinth_env import ToyLab

from envs.point2d_env import Point2DEnv


def make_toylab_dads_env():
    env = DADSCustomToyLabEnv()
    env = ObsAsOrderedDict(env)
    env = DADSWrapper(env)
    env = FilterObservation(env, filter_keys=["achieved_goal"])
    return FlattenObservation(env)


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


def make_point2d_dads_env():
    return DADSWrapper(Point2DEnv())


def make_fetch_pick_and_place_env():
    return _process_fetch_env(FixedGoalFetchPickAndPlaceEnv(reward_type="dense"))


def make_fetch_slide_env():
    return _process_fetch_env(FixedGoalFetchSlideEnv(reward_type="dense"))


def make_fetch_reach_env():
    return _process_fetch_env(DADSCustomFetchReachEnv(reward_type="dense"))


def _process_fetch_env(env: FetchEnv):
    env = DADSWrapper(env)
    env = FilterObservation(env, filter_keys=["observation"])
    return FlattenObservation(env)


def _get_goal_from_state_fetch(state: np.ndarray) -> np.ndarray:
    return state[..., 3:6] if _is_batch(state) else state[3:6]


class FixedGoalFetchPickAndPlaceEnv(FetchPickAndPlaceEnv):
    _fixed_goal = np.asarray([1.34803644, 0.71081931, 0.6831472])

    achieved_goal_from_state = staticmethod(_get_goal_from_state_fetch)

    def _sample_goal(self):
        return self._fixed_goal.copy()


class FixedGoalFetchSlideEnv(FetchSlideEnv):
    _fixed_goal = np.asarray([1.7, 0.75, 0.41401894])

    achieved_goal_from_state = staticmethod(_get_goal_from_state_fetch)

    def _sample_goal(self):
        return self._fixed_goal.copy()


class DADSCustomFetchReachEnv(FetchReachEnv):
    @staticmethod
    def achieved_goal_from_state(state: np.ndarray) -> np.ndarray:
        return state[..., :3] if _is_batch(state) else state[:3]


class DADSEnv(ABC, GoalEnv):
    goal: np.ndarray

    class OBS_TYPE(Enum):
        DYNAMICS_OBS = "DYNAMICS_OBS"
        FULL_OBS = "FULL_OBS"

    def to_dynamics_obs(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def achieved_goal_from_state(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DADSWrapper(Wrapper, DADSEnv):
    def compute_reward(self, achieved_goal, desired_goal, info):
        if info in DADSEnv.OBS_TYPE:
            return self._dads_reward(achieved_goal, desired_goal, obs_type=info)
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def _dads_reward(self, cur_obs, next_obs, obs_type: DADSEnv.OBS_TYPE):
        achieved_goal, next_achieved_goal = cur_obs, next_obs

        need_reduction = obs_type == DADSEnv.OBS_TYPE.FULL_OBS
        if need_reduction:
            achieved_goal = self.env.achieved_goal_from_state(cur_obs)
            next_achieved_goal = self.env.achieved_goal_from_state(next_obs)

        goal = np.broadcast_to(self.goal, achieved_goal.shape)
        reward = lambda achieved: self.env.compute_reward(achieved, goal, info=None)
        return reward(next_achieved_goal) - reward(achieved_goal)

    def to_dynamics_obs(self, obs: np.ndarray) -> np.ndarray:
        return self.env.achieved_goal_from_state(state=obs)

    def achieved_goal_from_state(self, state: np.ndarray) -> np.ndarray:
        return self.env.achieved_goal_from_state(state=state)
