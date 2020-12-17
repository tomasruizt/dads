import gym
import numpy as np
from gym import Wrapper, ObservationWrapper
from multi_goal.utils import get_updateable_scatter


class DictInfoWrapper(Wrapper):
    def __init__(self, env):
        assert hasattr(env, "is_success")
        super().__init__(env)

    def step(self, action):
        *step, _ = super().step(action)
        return (*step, dict(is_success=self.env.is_success()))


class PlotGoalWrapper(Wrapper):
    def __init__(self, env, goal_limit: float):
        super().__init__(env)
        assert hasattr(env, "_get_obs")
        self._plot = None
        self._goal_limit = goal_limit

    def render(self, mode='human', **kwargs):
        self.env.render(mode=mode, **kwargs)
        if self._plot is None:
            self._plot = get_updateable_scatter()
            l = self._goal_limit
            self._plot[1].plot([-l, l, l, -l, -l], [-l, -l, l, l, -l])
            self._plot[1].set_xlim((-1.1*l, 1.1*l))
            self._plot[1].set_ylim((-1.1*l, 1.1*l))
        fig, ax, scatter = self._plot
        obs = getattr(self.env, "_get_obs")()
        scatter(name="achieved_goal", pts=obs["achieved_goal"])
        scatter(name="desired_goal", pts=obs["desired_goal"])
        fig.canvas.draw()
        fig.canvas.flush_events()


class DenseGoalWrapper(Wrapper):
    def compute_reward(self, achieved_goal, desired_goal, info):
        return -np.linalg.norm(np.subtract(achieved_goal, desired_goal), axis=achieved_goal.ndim - 1)**2

    @staticmethod
    def achieved_goal_from_state(state: np.ndarray) -> np.ndarray:
        assert state.ndim <= 2, state.ndim
        is_batch = state.ndim == 2
        return state[..., :2] if is_batch else state[:2]


class DropGoalEnvsAbsoluteLocation(ObservationWrapper):
    """
    Requires GoalEnvs where:
    * The first 2 dims of the obs["observation"] are the current absolute location.
    * (achieved_goal, desired_goal) are 2d absolute locations
    """

    def __init__(self, env):
        super().__init__(env)
        goal_dim = env.observation_space["desired_goal"].shape[0]
        assert goal_dim == 2
        obs_dim = env.observation_space["observation"].shape[0]
        box = lambda d: gym.spaces.Box(-np.inf, np.inf, shape=(d, ))
        self.observation_space = gym.spaces.Dict(spaces=dict(
            observation=box(obs_dim - 2),
            achieved_goal=box(goal_dim),
            desired_goal=box(goal_dim)))

    def observation(self, observation: dict):
        abs_position = observation["observation"][:2]
        achieved_goal = np.zeros_like(abs_position)
        desired_goal = observation["desired_goal"] - abs_position
        obs_without_abs_position = observation["observation"][2:]
        return dict(observation=obs_without_abs_position,
                    desired_goal=desired_goal,
                    achieved_goal=achieved_goal)


def distance_to_goal(dict_obs: dict):
    return np.linalg.norm(np.subtract(dict_obs["achieved_goal"], dict_obs["desired_goal"]))