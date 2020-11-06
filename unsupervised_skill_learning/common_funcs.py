from abc import ABC
from typing import Callable

import numpy as np
import tensorflow as tf
from tf_agents.trajectories.time_step import TimeStep

from lib import py_tf_policy
from skill_slider import create_sliders_widget


def process_observation_given(obs, env_name: str, reduced_observation: int):
  def _shape_based_observation_processing(observation, dim_idx):
    if len(observation.shape) == 1:
      return observation[dim_idx:dim_idx + 1]
    elif len(observation.shape) == 2:
      return observation[:, dim_idx:dim_idx + 1]
    elif len(observation.shape) == 3:
      return observation[:, :, dim_idx:dim_idx + 1]

  # for consistent use
  if reduced_observation == 0:
    return obs

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
  if reduced_observation in [1, 5]:
    red_obs = [_shape_based_observation_processing(obs, 0)]
  # x-y plane
  elif reduced_observation in [2, 6]:
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
  elif reduced_observation in [4, 8]:
    if reduced_observation == 4 and 'DKittyPush' in env_name:
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
  elif reduced_observation == 3:
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

  if reduced_observation in [5, 6, 8]:
    red_obs += [
        _shape_based_observation_processing(obs,
                                            obs.shape[1] - idx)
        for idx in range(1, 5)
    ]

  if reduced_observation == 36 and 'DKitty' in env_name:
    red_obs = [
        _shape_based_observation_processing(obs, idx)
        for idx in range(qpos_dim)
    ]

  # x, y, z and the rotation quaternion
  if reduced_observation == 7 and env_name == 'HandBlock':
    red_obs = [
        _shape_based_observation_processing(obs, obs.shape[-1] - idx)
        for idx in range(1, 8)
    ][::-1]

  # the rotation quaternion
  if reduced_observation == 4 and env_name == 'HandBlock':
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
        return NotImplementedError

    def get_skill(self, ts: TimeStep):
        return NotImplementedError


class SliderSkillProvider(SkillProvider):
    def __init__(self, num_sliders: int):
        self._slider = create_sliders_widget(dim=num_sliders)

    def start_episode(self):
        pass

    def get_skill(self, ts: TimeStep):
        return self._slider.get_slider_values()


def evaluate_skill_provider(
        env,
        policy: py_tf_policy.PyTFPolicy,
        episode_length: int,
        hide_coords_fn: Callable,
        clip_action_fn: Callable,
        skill_provider):
    check_reward_fn(env)

    while True:
        timestep = env.reset()
        skill_provider.start_episode()
        for _ in range(episode_length):
            skill = skill_provider.get_skill(timestep)
            timestep = timestep._replace(observation=np.concatenate((timestep.observation, skill)))
            action = clip_action_fn(policy.action_mean(hide_coords_fn(timestep)))
            timestep = env.step(action)
            env.render("human")


def check_reward_fn(env):
    obs1 = env.observation_space.sample()
    obs2 = env.observation_space.sample()
    reward_dads = env.compute_reward(obs1, obs2, info="dads")

    to_goal = env.achieved_goal_from_state
    reward_classic = env.compute_reward(to_goal(obs1), to_goal(obs2), info=None)
    assert not np.isclose(reward_dads, reward_classic), f"Both rewards are equal to: {reward_dads}"
