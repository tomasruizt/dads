import numpy as np
from gym import Wrapper
from multi_goal.utils import get_updateable_scatter


class DictInfoWrapper(Wrapper):
    def step(self, action):
        *step, _ = super().step(action)
        return (*step, dict())


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
