import sys
from collections import OrderedDict
from functools import partial

import numpy as np
from gym import Wrapper, ObservationWrapper
from gym.envs.robotics import FetchPickAndPlaceEnv, FetchSlideEnv, FetchEnv, FetchReachEnv
from gym.wrappers import FilterObservation, FlattenObservation
from multi_goal.envs.toy_labyrinth_env import ToyLab

from envs.point2d_env import Point2DEnv


def make_toylab_dads_env():
    env = CustomToyLabEnv()
    env = ObsAsOrderedDict(env)
    env = DadsRewardWrapper(env)
    env = FilterObservation(env, filter_keys=["achieved_goal", "observation"])
    return FlattenObservation(env)


class CustomToyLabEnv(ToyLab):
    def __init__(self):
        super().__init__(max_episode_len=sys.maxsize, use_random_starting_pos=True)

    @staticmethod
    def achieved_goal_from_state(obs: np.ndarray) -> np.ndarray:
        return obs[..., :2] if is_batch(obs) else obs[:2]

    def compute_reward(self, achieved_obs, desired_obs, info):
        achieved_goal = self.achieved_goal_from_state(achieved_obs)
        desired_goal = self.achieved_goal_from_state(desired_obs)
        r = partial(super().compute_reward, info=info)
        if is_batch(achieved_obs):
            return np.asarray([r(a, d) for a, d in zip(achieved_obs, desired_obs)])
        return r(achieved_goal=achieved_goal, desired_goal=desired_goal)

    @property
    def goal(self):
        return self._goal_pos


def is_batch(x: np.ndarray) -> bool:
    return x.ndim == 2


class ObsAsOrderedDict(ObservationWrapper):
    def observation(self, observation):
        return OrderedDict(observation)


def make_point2d_dads_env():
    return DadsRewardWrapper(Point2DEnv())


def make_fetch_pick_and_place_env():
    return _process_fetch_env(FixedGoalFetchPickAndPlaceEnv(reward_type="dense"))


def make_fetch_slide_env():
    return _process_fetch_env(FixedGoalFetchSlideEnv(reward_type="dense"))


def make_fetch_reach_env():
    return _process_fetch_env(FixedGoalFetchReach(reward_type="dense"))


def _process_fetch_env(env: FetchEnv):
    env = DadsRewardWrapper(env)
    env = FilterObservation(env, filter_keys=["observation"])
    return FlattenObservation(env)


def _get_goal_from_state_fetch(state: np.ndarray) -> np.ndarray:
    return state[..., 3:6] if is_batch(state) else state[3:6]


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


class FixedGoalFetchReach(FetchReachEnv):
    _fixed_goal = np.asarray([1.34803644, 0.71081931, 0.6831472])

    @staticmethod
    def achieved_goal_from_state(obs: np.ndarray) -> np.ndarray:
        return obs[..., :3] if is_batch(obs) else obs[:3]

    def _sample_goal(self):
        return self._fixed_goal.copy()


class DadsRewardWrapper(Wrapper):
    def compute_reward(self, achieved_goal, desired_goal, info):
        if info == "dads":
            return self._dads_reward(achieved_goal, desired_goal)
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def _dads_reward(self, cur_state, next_state):
        achieved_goal = self.env.achieved_goal_from_state(cur_state)
        next_achieved_goal = self.env.achieved_goal_from_state(next_state)
        goal = np.broadcast_to(self.goal, achieved_goal.shape)
        reward = lambda achieved: self.env.compute_reward(achieved, goal, info=None)
        return reward(next_achieved_goal) - reward(achieved_goal)
