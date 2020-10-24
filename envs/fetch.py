from functools import partial
import numpy as np
from gym.envs.robotics import FetchPickAndPlaceEnv
from gym.wrappers import FilterObservation, FlattenObservation


class CustomFetchPickAndPlaceEnv(FetchPickAndPlaceEnv):
    _fixed_goal = np.asarray([1.34803644, 0.71081931, 0.6831472])

    def _sample_goal(self):
        return self._fixed_goal.copy()

    def compute_reward(self, cur_state, next_state, info=None):
        if info is not None:
            return super().compute_reward(cur_state, next_state, info)
        cur_block_pos = _block_pos(cur_state)
        next_block_pos = _block_pos(next_state)
        goal = np.broadcast_to(self._fixed_goal, cur_block_pos.shape)
        r = partial(super().compute_reward, goal=goal, info=None)
        return r(next_block_pos) - r(cur_block_pos)


def _block_pos(observation: np.ndarray) -> np.ndarray:
    is_batch = observation.ndim == 2
    if is_batch:
        return observation[..., 3:6]
    return observation[3:6]


def make_fetch_pick_and_place_env():
    filter_obs = partial(FilterObservation, filter_keys=["observation"])
    env = CustomFetchPickAndPlaceEnv(reward_type="dense")
    return FlattenObservation(filter_obs(env))
