# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from gym import utils, GoalEnv
import numpy as np
from gym.envs.mujoco import mujoco_env
from envs.gym_mujoco.custom_wrappers import DictInfoWrapper, PlotGoalWrapper, DenseGoalWrapper, \
    distance_to_goal


# pylint: disable=missing-docstring
class PointMassEnv(mujoco_env.MujocoEnv, utils.EzPickle):

  def __init__(self,
               target=None,
               wiggly_weight=0.,
               alt_xml=False,
               expose_velocity=True,
               expose_goal=True,
               use_simulator=False,
               model_path='point.xml',
               goal_limits=2):
    self._lims = goal_limits
    self._sample_target = target
    if self._sample_target is not None:
      self.goal = np.array([1.0, 1.0])

    self._expose_velocity = expose_velocity
    self._expose_goal = expose_goal
    self._use_simulator = use_simulator
    self._wiggly_weight = abs(wiggly_weight)
    self._wiggle_direction = +1 if wiggly_weight > 0. else -1

    xml_path = "envs/assets/"
    model_path = os.path.abspath(os.path.join(xml_path, model_path))

    if self._use_simulator:
      mujoco_env.MujocoEnv.__init__(self, model_path, 5)
    else:
      mujoco_env.MujocoEnv.__init__(self, model_path, 1)
    utils.EzPickle.__init__(self)

  def step(self, action):
    if self._use_simulator:
      self.do_simulation(action, self.frame_skip)
    else:
      force = 0.2 * action[0]
      rot = 1.0 * action[1]
      qpos = self.sim.data.qpos.flat.copy()
      qpos[2] += rot
      ori = qpos[2]
      dx = math.cos(ori) * force
      dy = math.sin(ori) * force
      qpos[0] = np.clip(qpos[0] + dx, -4*self._lims, 4*self._lims)
      qpos[1] = np.clip(qpos[1] + dy, -4*self._lims, 4*self._lims)
      qvel = self.sim.data.qvel.flat.copy()
      self.set_state(qpos, qvel)

    ob = self._get_obs()
    if self._sample_target is not None and self.goal is not None:
      reward = -np.linalg.norm(self.sim.data.qpos.flat[:2] - self.goal)**2
    else:
      reward = 0.

    if self._wiggly_weight > 0.:
      reward = (np.exp(-((-reward)**0.5))**(1. - self._wiggly_weight)) * (
          max(self._wiggle_direction * action[1], 0)**self._wiggly_weight)
    done = False
    return ob, reward, done, None

  def _get_obs(self):
    new_obs = [self.sim.data.qpos.flat]
    if self._expose_velocity:
      new_obs += [self.sim.data.qvel.flat]
    if self._expose_goal and self.goal is not None:
      new_obs += [self.goal]
    return np.concatenate(new_obs)

  def reset_model(self):
    qpos = self.init_qpos + np.append(
        self.np_random.uniform(low=-.2, high=.2, size=2),
        self.np_random.uniform(-np.pi, np.pi, size=1))
    qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * .01
    if self._sample_target is not None:
      self.goal = self._sample_target(qpos[:2])
    self.set_state(qpos, qvel)
    return self._get_obs()

  # only works when goal is not exposed
  def set_qpos(self, state):
    qvel = np.copy(self.sim.data.qvel.flat)
    self.set_state(state, qvel)

  def viewer_setup(self):
    self.viewer.cam.distance = self.model.stat.extent * 0.5


def PointMassGoalEnv() -> GoalEnv:
    goal_limit = 10
    env = PointMassAsGoalEnv(goal_limits=goal_limit)
    env = PlotGoalWrapper(env, goal_limit=goal_limit)
    env = DenseGoalWrapper(env)
    return DictInfoWrapper(env)


class PointMassAsGoalEnv(PointMassEnv):
    def __init__(self, goal_limits):
        super().__init__(expose_velocity=True, expose_goal=False,
                         target=lambda s: np.random.uniform(-goal_limits, goal_limits, size=2), goal_limits=goal_limits)

    def _get_obs(self):
        obs = super()._get_obs().astype(np.float32)
        assert len(obs) == 6, obs
        return dict(achieved_goal=obs[:2],
                    desired_goal=self.goal.astype(np.float32),
                    observation=obs)

    def is_success(self):
        dict_obs = self._get_obs()
        return float(distance_to_goal(dict_obs) < 0.5)


def _perfect_action(dict_obs):
    cur_orientation_rad = dict_obs["observation"][2] % (2*np.pi)
    cur_dir = np.asarray([np.cos(cur_orientation_rad), np.sin(cur_orientation_rad)])
    target_dir = dict_obs["desired_goal"] - dict_obs["achieved_goal"]
    target_orientation_rad = np.arctan2(target_dir[1], target_dir[0])
    if target_orientation_rad < 0:
        target_orientation_rad += 2*np.pi

    point_in_same_dir = 0 <= cur_dir @ target_dir
    move = 1.0 if point_in_same_dir else -1.0

    orientation_change = target_orientation_rad - cur_orientation_rad
    return [move, 0.1*orientation_change]


if __name__ == '__main__':
    env = PointMassGoalEnv()
    print(env.action_space)
    print(env.observation_space)
    print(env.reset())
    while True:
        obs = env.reset()
        for _ in range(150):
            env.render("human")
            action = _perfect_action(obs)
            obs, *_ = env.step(action)
