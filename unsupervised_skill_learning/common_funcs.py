from abc import ABC
from collections import deque
from typing import Callable, NamedTuple, Iterator
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
from itertools import islice
from tf_agents.trajectories.time_step import TimeStep

from envs.custom_envs import DADSEnv
from lib import py_tf_policy
from skill_slider import create_sliders_widget


def process_observation_given(obs, env_name: str, reduced_observation_size: int):
  if reduced_observation_size == 0:
      return obs

  def _shape_based_observation_processing(observation, dim_idx):
    if len(observation.shape) == 1:
      return observation[dim_idx:dim_idx + 1]
    elif len(observation.shape) == 2:
      return observation[:, dim_idx:dim_idx + 1]
    elif len(observation.shape) == 3:
      return observation[:, :, dim_idx:dim_idx + 1]

  # process observation for dynamics with reduced observation space
  if env_name == 'HalfCheetah-v1':
    qpos_dim = 9
  elif env_name == 'Ant-v1':
    qpos_dim = 15
  elif env_name == 'Humanoid-v1':
    qpos_dim = 26
  elif 'DKitty' in env_name:
    qpos_dim = 36

  # x-axis
  if reduced_observation_size in [1, 5]:
    red_obs = [_shape_based_observation_processing(obs, 0)]
  # x-y plane
  elif reduced_observation_size in [2, 6]:
    if env_name == 'Ant-v1' or 'DKitty' in env_name or 'DClaw' in env_name:
      red_obs = [
          _shape_based_observation_processing(obs, 0),
          _shape_based_observation_processing(obs, 1)
      ]
    else:
      red_obs = [
          _shape_based_observation_processing(obs, 0),
          _shape_based_observation_processing(obs, qpos_dim)
      ]
  # x-y plane, x-y velocities
  elif reduced_observation_size in [4, 8]:
    if reduced_observation_size == 4 and 'DKittyPush' in env_name:
      # position of the agent + relative position of the box
      red_obs = [
          _shape_based_observation_processing(obs, 0),
          _shape_based_observation_processing(obs, 1),
          _shape_based_observation_processing(obs, 3),
          _shape_based_observation_processing(obs, 4)
      ]
    elif env_name in ['Ant-v1']:
      red_obs = [
          _shape_based_observation_processing(obs, 0),
          _shape_based_observation_processing(obs, 1),
          _shape_based_observation_processing(obs, qpos_dim),
          _shape_based_observation_processing(obs, qpos_dim + 1)
      ]

  # (x, y, orientation), works only for ant, point_mass
  elif reduced_observation_size == 3:
    if env_name in ['Ant-v1', 'point_mass']:
      red_obs = [
          _shape_based_observation_processing(obs, 0),
          _shape_based_observation_processing(obs, 1),
          _shape_based_observation_processing(obs,
                                              obs.shape[1] - 1)
      ]
    # x, y, z of the center of the block
    elif env_name in ['HandBlock']:
      red_obs = [
          _shape_based_observation_processing(obs,
                                              obs.shape[-1] - 7),
          _shape_based_observation_processing(obs,
                                              obs.shape[-1] - 6),
          _shape_based_observation_processing(obs,
                                              obs.shape[-1] - 5)
      ]

  if reduced_observation_size in [5, 6, 8]:
    red_obs += [
        _shape_based_observation_processing(obs,
                                            obs.shape[1] - idx)
        for idx in range(1, 5)
    ]

  if reduced_observation_size == 36 and 'DKitty' in env_name:
    red_obs = [
        _shape_based_observation_processing(obs, idx)
        for idx in range(qpos_dim)
    ]

  # x, y, z and the rotation quaternion
  if reduced_observation_size == 7 and env_name == 'HandBlock':
    red_obs = [
        _shape_based_observation_processing(obs, obs.shape[-1] - idx)
        for idx in range(1, 8)
    ][::-1]

  # the rotation quaternion
  if reduced_observation_size == 4 and env_name == 'HandBlock':
    red_obs = [
        _shape_based_observation_processing(obs, obs.shape[-1] - idx)
        for idx in range(1, 5)
    ][::-1]

  if isinstance(obs, np.ndarray):
    input_obs = np.concatenate(red_obs, axis=len(obs.shape) - 1)
  elif isinstance(obs, tf.Tensor):
    input_obs = tf.concat(red_obs, axis=len(obs.shape) - 1)
  return input_obs


def hide_coordinates(time_step, first_n: int):
  if first_n > 0:
    sans_coords = time_step.observation[first_n:]
    return time_step._replace(observation=sans_coords)
  return time_step


def clip(x, low: float, high: float):
    return np.clip(x, a_min=low, a_max=high)


class SkillProvider(ABC):
    def start_episode(self):
        raise NotImplementedError

    def get_skill(self, ts: TimeStep) -> np.ndarray:
        raise NotImplementedError


class RandomSkillProvider(SkillProvider):
    def __init__(self, skill_dim: int):
        self._skill_dim = skill_dim

    def start_episode(self):
        pass

    def get_skill(self, ts: TimeStep) -> np.ndarray:
        return np.random.uniform(-1, 1, self._skill_dim)


class NullSkillProvider(SkillProvider):
    def __init__(self, skill_dim: int):
        self._skill_dim = skill_dim

    def start_episode(self):
        pass

    def get_skill(self, ts: TimeStep):
        return np.zeros(self._skill_dim)


class SliderSkillProvider(SkillProvider):
    def __init__(self, num_sliders: int):
        self._slider = create_sliders_widget(dim=num_sliders)

    def start_episode(self):
        pass

    def get_skill(self, ts: TimeStep):
        return self._slider.get_slider_values()


class DADSStep(NamedTuple):
    ts: TimeStep
    skill: np.ndarray
    ts_p1: TimeStep
    goal: np.ndarray


def evaluate_skill_provider_loop(
        env,
        policy: py_tf_policy.PyTFPolicy,
        episode_length: int,
        hide_coords_fn: Callable,
        clip_action_fn: Callable,
        skill_provider,
        num_episodes=sys.maxsize,
        render_env=False) -> Iterator[DADSStep]:

    for _ in range(num_episodes):
        timestep = env.reset()
        skill_provider.start_episode()
        for _ in range(episode_length):
            skill = skill_provider.get_skill(timestep)
            timestep_pskill: TimeStep = timestep._replace(observation=np.concatenate((timestep.observation, skill)))
            action = clip_action_fn(policy.action_mean(hide_coords_fn(timestep_pskill)))
            next_timestep = env.step(action)
            yield DADSStep(ts=timestep, skill=skill, ts_p1=next_timestep, goal=env.goal.copy())
            timestep = next_timestep
            if render_env:
                env.render("human")


def check_reward_fn(env: DADSEnv):
    obs1 = env.observation_space.sample()
    obs2 = env.observation_space.sample()
    reward_dads = env.compute_reward(obs1, obs2, info=DADSEnv.OBS_TYPE.FULL_OBS)

    to_goal = env.achieved_goal_from_state
    reward_classic = env.compute_reward(to_goal(obs1), to_goal(obs2), info=None)
    assert not np.isclose(reward_dads, reward_classic), f"Both rewards are equal to: {reward_dads}"


def consume(iterator):
    deque(iterator, maxlen=0)


class PlanViz:
    def __init__(self):
        self._inited = False

    def _initialize(self):
        plt.ion()
        self._fig, (self._ax, self._skills_ax) = plt.subplots(2, 1)
        self._cur_pos_scatter = self._ax.scatter(0, 0, c="orange")
        self._prev_pos_scatter = self._ax.scatter(0, 0, c="brown")
        self._goal_scatter = self._ax.scatter(0, 0, c="red")
        self._highlight_pt_scatter = self._ax.scatter(0, 0, c="pink")
        self._pts_scatter = self._ax.scatter([None], [None], c=[0], cmap="viridis")
        self._direction = self._ax.quiver([0, 0], [1, 1])

        self._skills_scatter = self._skills_ax.scatter([None], [None], c=[0], cmap="viridis")
        self._sel_skill_scatter = self._skills_ax.scatter(0, 0, c="pink")
        self._skills_ax.set_xlim((-1, 1))
        self._skills_ax.set_ylim((-1, 1))

        self._inited = True

    def update(self, prev_pos: np.ndarray, cur_pos: np.ndarray, goal: np.ndarray,
               highlight_pt: np.ndarray, pts: np.ndarray, rewards: np.ndarray,
               candidate_skills: np.ndarray, selected_skill: np.ndarray) -> None:
        if not self._inited:
            self._initialize()
        self._direction.remove()
        self._direction = self._ax.quiver(*cur_pos, *(highlight_pt - cur_pos), angles="xy", zorder=-1)
        self._pts_scatter.remove()
        self._pts_scatter = self._ax.scatter(*pts.T, c=rewards, cmap="viridis", zorder=-1)
        self._cur_pos_scatter.set_offsets(cur_pos)
        self._prev_pos_scatter.set_offsets(prev_pos)
        self._goal_scatter.set_offsets(goal)
        self._highlight_pt_scatter.set_offsets(highlight_pt)
        self._ax.set_xlim(np.asarray([-0.25, 0.25]) + 1.3)
        self._ax.set_ylim(np.asarray([-0.4, 0.4]) + 0.75)

        self._skills_scatter.remove()
        self._skills_scatter = self._skills_ax.scatter(*candidate_skills[..., :2].T, c=rewards, cmap="viridis")
        self._sel_skill_scatter.set_offsets(selected_skill[:2])

        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


def grouper(n, iterable):
    """
    >>> list(grouper(3, 'ABCDEFG'))
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]
    """
    iterable = iter(iterable)
    return iter(lambda: list(islice(iterable, n)), [])
