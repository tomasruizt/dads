import numpy as np
from gym import Wrapper
from gym.envs.robotics import FetchPickAndPlaceEnv, FetchSlideEnv, FetchEnv, FetchReachEnv
from gym.wrappers import FilterObservation, FlattenObservation

from envs.point2d_env import Point2DEnv


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
    is_batch = state.ndim == 2
    if is_batch:
        return state[..., 3:6]
    return state[3:6]


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
        is_batch = obs.ndim is 2
        if is_batch:
            return obs[..., :3]
        return obs[:3]

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


