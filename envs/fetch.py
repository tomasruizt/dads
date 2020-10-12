from functools import partial
import numpy as np
from gym.envs.robotics import FetchPickAndPlaceEnv
from gym.wrappers import FilterObservation, FlattenObservation


class FixedGoalFetchPickAndPlaceEnv(FetchPickAndPlaceEnv):
    _fixed_goal = np.asarray([1.34803644, 0.71081931, 0.6831472])

    def _sample_goal(self):
        return self._fixed_goal.copy()


def make_fetch_pick_and_place_env():
    filter_obs = partial(FilterObservation, filter_keys=["observation", "achieved_goal"])
    env = FixedGoalFetchPickAndPlaceEnv(reward_type="dense")
    return FlattenObservation(filter_obs(env))
