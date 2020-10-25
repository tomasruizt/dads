from functools import partial
import numpy as np
from gym import Wrapper
from gym.envs.robotics import FetchPickAndPlaceEnv, FetchSlideEnv, FetchEnv
from gym.wrappers import FilterObservation, FlattenObservation


def make_fetch_pick_and_place_env():
    return _process_fetch_env(FixedGoalFetchPickAndPlaceEnv(reward_type="dense"))


def make_fetch_slide_env():
    return _process_fetch_env(FixedGoalFetchSlideEnv(reward_type="dense"))


def _process_fetch_env(env: FetchEnv):
    env = DadsRewardWrapper(env)
    env = FilterObservation(env, filter_keys=["observation"])
    return FlattenObservation(env)


class FixedGoalFetchPickAndPlaceEnv(FetchPickAndPlaceEnv):
    _fixed_goal = np.asarray([1.34803644, 0.71081931, 0.6831472])

    def _sample_goal(self):
        return self._fixed_goal.copy()


class FixedGoalFetchSlideEnv(FetchSlideEnv):
    _fixed_goal = np.asarray([1.7, 0.75, 0.41401894])

    def _sample_goal(self):
        return self._fixed_goal.copy()


class DadsRewardWrapper(Wrapper):
    def compute_reward(self, achieved_goal, desired_goal, info):
        if info == "dads":
            return self._dads_reward(achieved_goal, desired_goal)
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def _dads_reward(self, cur_state, next_state):
        cur_block_pos = _block_pos(cur_state)
        next_block_pos = _block_pos(next_state)
        goal = np.broadcast_to(self.goal, cur_block_pos.shape)
        r = partial(self.env.compute_reward, goal=goal, info=None)
        return r(next_block_pos) - r(cur_block_pos)


def _block_pos(observation: np.ndarray) -> np.ndarray:
    is_batch = observation.ndim == 2
    if is_batch:
        return observation[..., 3:6]
    return observation[3:6]
