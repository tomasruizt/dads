from functools import partial

from gym.envs.robotics import FetchPickAndPlaceEnv
from gym.wrappers import TimeLimit, FilterObservation, FlattenObservation


def make_fetch_pick_and_place_env():
    filter_obs = partial(FilterObservation, filter_keys=["observation", "achieved_goal"])
    env = FetchPickAndPlaceEnv()
    return TimeLimit(max_episode_steps=100, env=FlattenObservation(filter_obs(env)))
