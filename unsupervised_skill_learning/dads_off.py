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

import os
import io
from functools import partial
from typing import List, Generator, Sequence, Callable
import gtimer as gt
from absl import flags, logging

import sys

from tf_agents.trajectories.time_step import TimeStep

from common_funcs import DADSStep, grouper, check_reward_fn, NullSkillProvider, \
    RandomSkillProvider
from custom_mppi import MPPISkillProvider
from density_estimation import DensityEstimator
from envs.custom_envs import make_fetch_pick_and_place_env, make_fetch_slide_env, \
    make_point2d_dads_env, make_fetch_reach_env, DADSEnv, make_fetch_push_env
from lib.simple_buffer import SimpleBuffer, Transition
from skill_dynamics import SkillDynamics, l2
from unsupervised_skill_learning.common_funcs import process_observation_given, \
    hide_coordinates, clip, SliderSkillProvider, evaluate_skill_provider_loop, \
    SkillProvider, consume
from unsupervised_skill_learning.mppi import mppi_next_skill_loop

sys.path.append(os.path.abspath('./'))

import matplotlib
import seaborn as sns
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.environments import suite_mujoco
from tf_agents.trajectories import time_step as ts
from tf_agents.environments.suite_gym import wrap_env
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import ou_noise_policy
from tf_agents.trajectories import policy_step
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.utils import nest_utils

import dads_agent

from envs import skill_wrapper
from envs import video_wrapper
from envs.gym_mujoco import ant
from envs.gym_mujoco import half_cheetah
from envs.gym_mujoco import humanoid
from envs.gym_mujoco import point_mass

from envs import dclaw
from envs import dkitty_redesign
from envs import hand_block

from lib import py_tf_policy

FLAGS = flags.FLAGS
nest = tf.nest

# general hyperparameters
flags.DEFINE_string('logdir', '~/tmp/dads', 'Directory for saving experiment data')
flags.DEFINE_string('experiment_name', default='', help="Optional experiment name to postfix the logdir")
flags.DEFINE_integer("seed", 0, "Seed of this experiment")

# environment hyperparameters
flags.DEFINE_string('environment', 'point_mass', 'Name of the environment')
flags.DEFINE_integer('max_env_steps', 200,
                     'Maximum number of steps in one episode')
flags.DEFINE_integer('max_env_steps_eval', 200, 'Steps per episode when evaluating')
flags.DEFINE_integer('reduced_observation_size', 0,
                     'Predict dynamics in a reduced observation space')
flags.DEFINE_boolean('use_state_space_reduction', True,
                     "Reduce the state space to the goal space in the mutual information objective")
flags.DEFINE_integer(
    'min_steps_before_resample', 50,
    'Minimum number of steps to execute before resampling skill')
flags.DEFINE_float('resample_prob', 0.,
                   'Creates stochasticity timesteps before resampling skill')

# need to set save_model and save_freq
flags.DEFINE_string(
    'save_model', None,
    'Name to save the model with, None implies the models are not saved.')
flags.DEFINE_integer('save_freq', 100, 'Saving frequency for checkpoints')
flags.DEFINE_string(
    'vid_name', None,
    'Base name for videos being saved, None implies videos are not recorded')
flags.DEFINE_integer('record_freq', 100,
                     'Video recording frequency within the training loop')

# final evaluation after training is done
flags.DEFINE_integer('run_final_eval', 0, 'Evaluate learnt skills')

# evaluation type
flags.DEFINE_integer('num_evals', 0, 'Number of skills to evaluate')
flags.DEFINE_integer('deterministic_eval', 0,
                  'Evaluate all skills, only works for discrete skills')
flags.DEFINE_integer('per_skill_evals', 1, 'Number of evaluation runs for each skill')

# training
flags.DEFINE_integer('run_train', 0, 'Train the agent')
flags.DEFINE_integer('num_epochs', 500, 'Number of training epochs')

# skill latent space
flags.DEFINE_integer('num_skills', 2, 'Number of skills to learn')
flags.DEFINE_string('skill_type', 'cont_uniform',
                    'Type of skill and the prior over it')
# network size hyperparameter
flags.DEFINE_integer(
    'hidden_layer_size', 512,
    'Hidden layer size, shared by actors, critics and dynamics')

# reward structure
flags.DEFINE_integer(
    'random_skills', 0,
    'Number of skills to sample randomly for approximating mutual information')

# optimization hyperparameters
flags.DEFINE_integer('replay_buffer_capacity', int(1e6),
                     'Capacity of the replay buffer')
flags.DEFINE_integer(
    'clear_buffer_every_iter', 0,
    'Clear replay buffer every iteration to simulate on-policy training, use larger collect steps and train-steps'
)
flags.DEFINE_integer(
    'initial_collect_steps', 2000,
    'Steps collected initially before training to populate the buffer')
flags.DEFINE_integer('collect_steps', 200, 'Steps collected per agent update')

# relabelling
flags.DEFINE_string('agent_relabel_type', None,
                    'Type of skill relabelling used for agent')
flags.DEFINE_integer(
    'train_skill_dynamics_on_policy', 0,
    'Train skill-dynamics on policy data, while agent train off-policy')
flags.DEFINE_string('skill_dynamics_relabel_type', None,
                    'Type of skill relabelling used for skill-dynamics')
flags.DEFINE_integer(
    'num_samples_for_relabelling', 100,
    'Number of samples from prior for relabelling the current skill when using policy relabelling'
)
flags.DEFINE_float(
    'is_clip_eps', 0.,
    'PPO style clipping epsilon to constrain importance sampling weights to (1-eps, 1+eps)'
)
flags.DEFINE_float(
    'action_clipping', 1.,
    'Clip actions to (-eps, eps) per dimension to avoid difficulties with tanh')
flags.DEFINE_integer('debug_skill_relabelling', 0,
                     'analysis of skill relabelling')
flags.DEFINE_boolean("use_dynamics_uniform_resampling", False,
                     "Train dynamics by resampling uniformly over the possible state deltas.")

# skill dynamics optimization hyperparamaters
flags.DEFINE_integer('skill_dyn_train_steps', 8,
                     'Number of discriminator train steps on a batch of data')
flags.DEFINE_float('skill_dynamics_lr', 3e-4,
                   'Learning rate for increasing the log-likelihood')
flags.DEFINE_integer('skill_dyn_batch_size', 256,
                     'Batch size for discriminator updates')
# agent optimization hyperparameters
flags.DEFINE_integer('agent_batch_size', 256, 'Batch size for agent updates')
flags.DEFINE_integer('agent_train_steps', 128,
                     'Number of update steps per iteration')
flags.DEFINE_float('agent_lr', 3e-4, 'Learning rate for the agent')

# SAC hyperparameters
flags.DEFINE_float('agent_entropy', 0.1, 'Entropy regularization coefficient')
flags.DEFINE_float('agent_gamma', 0.99, 'Reward discount factor')
flags.DEFINE_string(
    'collect_policy', 'default',
    'Can use the OUNoisePolicy to collect experience for better exploration')

# skill-dynamics hyperparameters
flags.DEFINE_string(
    'graph_type', 'default',
    'process skill input separately for more representational power')
flags.DEFINE_integer('num_components', 4,
                     'Number of components for Mixture of Gaussians')
flags.DEFINE_integer('fix_variance', 1,
                     'Fix the variance of output distribution')
flags.DEFINE_integer('normalize_data', 1, 'Maintain running averages')

# debug
flags.DEFINE_integer('debug', 0, 'Creates extra summaries')
flags.DEFINE_boolean('manual_control_mode', False, 'Pop a slider to control the skills manually.')
flags.DEFINE_boolean('no_control_mode', False, 'Viz behavior and metrics under no control')
flags.DEFINE_boolean('random_skills_control_mode', False, 'Viz behavior under random skills')

# DKitty
flags.DEFINE_integer('expose_last_action', 1, 'Add the last action to the observation')
flags.DEFINE_integer('expose_upright', 1, 'Add the upright angle to the observation')
flags.DEFINE_float('upright_threshold', 0.9, 'Threshold before which the DKitty episode is terminated')
flags.DEFINE_float('robot_noise_ratio', 0.05, 'Noise ratio for robot joints')
flags.DEFINE_float('root_noise_ratio', 0.002, 'Noise ratio for root position')
flags.DEFINE_float('scale_root_position', 1, 'Multiply the root coordinates the magnify the change')
flags.DEFINE_integer('run_on_hardware', 0, 'Flag for hardware runs')
flags.DEFINE_float('randomize_hfield', 0.0, 'Randomize terrain for better DKitty transfer')
flags.DEFINE_integer('observation_omission_size', 2, 'Dimensions to be omitted from policy input')

# Manipulation Environments
flags.DEFINE_integer('randomized_initial_distribution', 1, 'Fix the initial distribution or not')
flags.DEFINE_float('horizontal_wrist_constraint', 1.0, 'Action space constraint to restrict horizontal motion of the wrist')
flags.DEFINE_float('vertical_wrist_constraint', 1.0, 'Action space constraint to restrict vertical motion of the wrist')

# MPC hyperparameters
flags.DEFINE_integer('planning_horizon', 1, 'Number of primitives to plan in the future')
flags.DEFINE_integer('primitive_horizon', 1, 'Horizon for every primitive')
flags.DEFINE_integer('num_candidate_sequences', 50, 'Number of candidates sequence sampled from the proposal distribution')
flags.DEFINE_integer('refine_steps', 10, 'Number of optimization steps')
flags.DEFINE_float('mppi_gamma', 10.0, 'MPPI weighting hyperparameter')
flags.DEFINE_string('prior_type', 'normal', 'Uniform or Gaussian prior for candidate skill(s)')
flags.DEFINE_float('smoothing_beta', 0.9, 'Smooth candidate skill sequences used')
flags.DEFINE_integer('top_primitives', 5, 'Optimization parameter when using uniform prior (CEM style)')

# global variables for this script
observation_omit_size = 0
sample_count = 0
iter_count = 0
episode_size_buffer = []
episode_return_buffer = []


# add a flag for state dependent std
def _normal_projection_net(action_spec, init_means_output_factor=0.1):
  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      mean_transform=None,
      state_dependent_std=True,
      init_means_output_factor=init_means_output_factor,
      std_transform=sac_agent.std_clip_transform,
      scale_distribution=True)


custom_envs_ctors = dict(
    point2d=make_point2d_dads_env,
    reach=make_fetch_reach_env,
    push=make_fetch_push_env,
    pickandplace=make_fetch_pick_and_place_env,
    slide=make_fetch_slide_env,
)


def get_environment(env_name='point_mass'):
    env = get_env_without_postprocess(env_name=env_name)
    return add_to_dynamics_obs_fn(env)


def add_to_dynamics_obs_fn(env):
    if hasattr(env, "to_dynamics_obs"):
        return env

    def to_dynamics_obs(obs):
        return process_observation_given(
            obs=obs,
            env_name=FLAGS.environment,
            reduced_observation_size=FLAGS.reduced_observation_size)
    env.to_dynamics_obs = to_dynamics_obs
    return env


def get_env_without_postprocess(env_name: str):
  global observation_omit_size

  simple_name = env_name.replace("_goal", "")
  if simple_name in custom_envs_ctors:
      env = custom_envs_ctors[simple_name](use_state_space_reduction=FLAGS.use_state_space_reduction)
      if env_name.endswith("_goal"):
          return wrap_env(env, max_episode_steps=FLAGS.max_env_steps)
  elif env_name == 'Ant-v1':
    env = ant.AntEnv(
        expose_all_qpos=True,
        task='motion')
    observation_omit_size = 2
  elif env_name == 'Ant-v1_goal':
    observation_omit_size = 2
    return wrap_env(
        ant.AntEnv(
            task='goal',
            goal=np.array([10., 10.]),
            expose_all_qpos=True),
        max_episode_steps=FLAGS.max_env_steps)
  elif env_name == 'Ant-v1_foot_sensor':
    env = ant.AntEnv(
        expose_all_qpos=True,
        model_path='ant_footsensor.xml',
        expose_foot_sensors=True)
    observation_omit_size = 2
  elif env_name == 'HalfCheetah-v1':
    env = half_cheetah.HalfCheetahEnv(expose_all_qpos=True, task='motion')
    observation_omit_size = 1
  elif env_name == 'Humanoid-v1':
    env = humanoid.HumanoidEnv(expose_all_qpos=True)
    observation_omit_size = 2
  elif env_name == 'point_mass':
    env = point_mass.PointMassEnv(expose_goal=False, expose_velocity=False)
    observation_omit_size = 2
  elif env_name == 'DClaw':
    env = dclaw.DClawTurnRandom()
    observation_omit_size = FLAGS.observation_omission_size
  elif env_name == 'DClaw_randomized':
    env = dclaw.DClawTurnRandomDynamics()
    observation_omit_size = FLAGS.observation_omission_size
  elif env_name == 'DKitty_redesign':
    env = dkitty_redesign.BaseDKittyWalk(
        expose_last_action=FLAGS.expose_last_action,
        expose_upright=FLAGS.expose_upright,
        robot_noise_ratio=FLAGS.robot_noise_ratio,
        upright_threshold=FLAGS.upright_threshold)
    observation_omit_size = FLAGS.observation_omission_size
  elif env_name == 'DKitty_randomized':
    env = dkitty_redesign.DKittyRandomDynamics(
        randomize_hfield=FLAGS.randomize_hfield,
        expose_last_action=FLAGS.expose_last_action,
        expose_upright=FLAGS.expose_upright,
        robot_noise_ratio=FLAGS.robot_noise_ratio,
        upright_threshold=FLAGS.upright_threshold)
    observation_omit_size = FLAGS.observation_omission_size
  elif env_name == 'HandBlock':
    observation_omit_size = 0
    env = hand_block.HandBlockCustomEnv(
        horizontal_wrist_constraint=FLAGS.horizontal_wrist_constraint,
        vertical_wrist_constraint=FLAGS.vertical_wrist_constraint,
        randomize_initial_position=bool(FLAGS.randomized_initial_distribution),
        randomize_initial_rotation=bool(FLAGS.randomized_initial_distribution))
  else:
    # note this is already wrapped, no need to wrap again
    env = suite_mujoco.load(env_name)
  return env


def hide_coords(time_step):
    global observation_omit_size
    return hide_coordinates(time_step=time_step, first_n=observation_omit_size)


def relabel_skill(trajectory_sample,
                  relabel_type=None,
                  cur_policy=None,
                  cur_skill_dynamics=None,
                  env=None):
  global observation_omit_size
  if relabel_type is None or ('importance_sampling' in relabel_type and
                              FLAGS.is_clip_eps <= 1.0):
    return trajectory_sample, None

  # trajectory.to_transition, but for numpy arrays
  next_trajectory = nest.map_structure(lambda x: x[:, 1:], trajectory_sample)
  trajectory = nest.map_structure(lambda x: x[:, :-1], trajectory_sample)
  action_steps = policy_step.PolicyStep(
      action=trajectory.action, state=(), info=trajectory.policy_info)
  time_steps = ts.TimeStep(
      trajectory.step_type,
      reward=nest.map_structure(np.zeros_like, trajectory.reward),  # unknown
      discount=np.zeros_like(trajectory.discount),  # unknown
      observation=trajectory.observation)
  next_time_steps = ts.TimeStep(
      step_type=trajectory.next_step_type,
      reward=trajectory.reward,
      discount=trajectory.discount,
      observation=next_trajectory.observation)
  time_steps, action_steps, next_time_steps = nest.map_structure(
      lambda t: np.squeeze(t, axis=1),
      (time_steps, action_steps, next_time_steps))

  # just return the importance sampling weights for the given batch
  if 'importance_sampling' in relabel_type:
    old_log_probs = policy_step.get_log_probability(action_steps.info)
    is_weights = []
    for idx in range(time_steps.observation.shape[0]):
      cur_time_step = nest.map_structure(lambda x: x[idx:idx + 1], time_steps)
      cur_time_step = cur_time_step._replace(
          observation=cur_time_step.observation[:, observation_omit_size:])
      old_log_prob = old_log_probs[idx]
      cur_log_prob = cur_policy.log_prob(cur_time_step,
                                         action_steps.action[idx:idx + 1])[0]
      is_weights.append(
          np.clip(
              np.exp(cur_log_prob - old_log_prob), 1. / FLAGS.is_clip_eps,
              FLAGS.is_clip_eps))

    is_weights = np.array(is_weights)
    if relabel_type == 'normalized_importance_sampling':
      is_weights = is_weights / is_weights.mean()

    return trajectory_sample, is_weights

  new_observation = np.zeros(time_steps.observation.shape)
  for idx in range(time_steps.observation.shape[0]):
    alt_time_steps = nest.map_structure(
        lambda t: np.stack([t[idx]] * FLAGS.num_samples_for_relabelling),
        time_steps)

    # sample possible skills for relabelling from the prior
    if FLAGS.skill_type == 'cont_uniform':
      # always ensure that the original skill is one of the possible option for relabelling skills
      alt_skills = np.concatenate([
          np.random.uniform(
              low=-1.0,
              high=1.0,
              size=(FLAGS.num_samples_for_relabelling - 1, FLAGS.num_skills)),
          alt_time_steps.observation[:1, -FLAGS.num_skills:]
      ])

    # choose the skill which gives the highest log-probability to the current action
    if relabel_type == 'policy':
      cur_action = np.stack([action_steps.action[idx, :]] *
                            FLAGS.num_samples_for_relabelling)
      alt_time_steps = alt_time_steps._replace(
          observation=np.concatenate([
              alt_time_steps
              .observation[:,
                           observation_omit_size:-FLAGS.num_skills], alt_skills
          ],
                                     axis=1))
      action_log_probs = cur_policy.log_prob(alt_time_steps, cur_action)
      if FLAGS.debug_skill_relabelling:
        print('\n action_log_probs analysis----', idx,
              time_steps.observation[idx, -FLAGS.num_skills:])
        print('number of skills with higher log-probs:',
              np.sum(action_log_probs >= action_log_probs[-1]))
        print('Skills with log-probs higher than actual skill:')
        skill_dist = []
        for skill_idx in range(FLAGS.num_samples_for_relabelling):
          if action_log_probs[skill_idx] >= action_log_probs[-1]:
            print(alt_skills[skill_idx])
            skill_dist.append(
                np.linalg.norm(alt_skills[skill_idx] - alt_skills[-1]))
        print('average distance of skills with higher-log-prob:',
              np.mean(skill_dist))
      max_skill_idx = np.argmax(action_log_probs)

    # choose the skill which gets the highest log-probability under the dynamics posterior
    elif relabel_type == 'dynamics_posterior':
      cur_observations = alt_time_steps.observation[:, :-FLAGS.num_skills]
      next_observations = np.stack(
          [next_time_steps.observation[idx, :-FLAGS.num_skills]] *
          FLAGS.num_samples_for_relabelling)

      # max over posterior log probability is exactly the max over log-prob of transitin under skill-dynamics
      posterior_log_probs = cur_skill_dynamics.get_log_prob(
          env.to_dynamics_obs(cur_observations), alt_skills,
          env.to_dynamics_obs(next_observations))
      if FLAGS.debug_skill_relabelling:
        print('\n dynamics_log_probs analysis----', idx,
              time_steps.observation[idx, -FLAGS.num_skills:])
        print('number of skills with higher log-probs:',
              np.sum(posterior_log_probs >= posterior_log_probs[-1]))
        print('Skills with log-probs higher than actual skill:')
        skill_dist = []
        for skill_idx in range(FLAGS.num_samples_for_relabelling):
          if posterior_log_probs[skill_idx] >= posterior_log_probs[-1]:
            print(alt_skills[skill_idx])
            skill_dist.append(
                np.linalg.norm(alt_skills[skill_idx] - alt_skills[-1]))
        print('average distance of skills with higher-log-prob:',
              np.mean(skill_dist))

      max_skill_idx = np.argmax(posterior_log_probs)

    # make the new observation with the relabelled skill
    relabelled_skill = alt_skills[max_skill_idx]
    new_observation[idx] = np.concatenate(
        [time_steps.observation[idx, :-FLAGS.num_skills], relabelled_skill])

  traj_observation = np.copy(trajectory_sample.observation)
  traj_observation[:, 0] = new_observation
  new_trajectory_sample = trajectory_sample._replace(
      observation=traj_observation)

  return new_trajectory_sample, None


def collect_experience(py_env,
                       time_step,
                       collect_policy,
                       replay_buffer: SimpleBuffer,
                       num_steps=1):

  episode_sizes = []
  extrinsic_reward = []
  step_idx = 0
  cur_return = 0.
  for step_idx in range(num_steps):
    if time_step.is_last():
      episode_sizes.append(step_idx)
      extrinsic_reward.append(cur_return)
      cur_return = 0.

    action_step = collect_policy.action(hide_coords(time_step))

    if FLAGS.action_clipping < 1.:
      action_step = action_step._replace(
          action=np.clip(action_step.action, -FLAGS.action_clipping,
                         FLAGS.action_clipping))

    if FLAGS.skill_dynamics_relabel_type is not None and 'importance_sampling' in FLAGS.skill_dynamics_relabel_type and FLAGS.is_clip_eps > 1.0:
      cur_action_log_prob = collect_policy.log_prob(
          nest_utils.batch_nested_array(hide_coords(time_step)),
          np.expand_dims(action_step.action, 0))
      action_step = action_step._replace(
          info=policy_step.set_log_probability(action_step.info,
                                               cur_action_log_prob))

    next_time_step = py_env.step(action_step.action)
    cur_return += next_time_step.reward
    if (not time_step.is_last()) and (not next_time_step.is_last()):
        transition = Transition(s=time_step.observation,
                                a=action_step.action,
                                s_next=next_time_step.observation)
        replay_buffer.add(transition)

    time_step = next_time_step

  # carry-over calculation for the next collection cycle
  episode_sizes.append(step_idx + 1)
  extrinsic_reward.append(cur_return)
  for idx in range(1, len(episode_sizes)):
    episode_sizes[-idx] -= episode_sizes[-idx - 1]

  return time_step, {
      'episode_sizes': episode_sizes,
      'episode_return': extrinsic_reward
  }


def run_on_env(env,
               policy,
               dynamics=None,
               predict_trajectory_steps=0,
               return_data=False,
               close_environment=True):
  time_step = env.reset()
  data = []

  if not return_data:
    extrinsic_reward = []
  while not time_step.is_last():
    action_step = policy.action(hide_coords(time_step))
    if FLAGS.action_clipping < 1.:
      action_step = action_step._replace(
          action=np.clip(action_step.action, -FLAGS.action_clipping,
                         FLAGS.action_clipping))

    env_action = action_step.action
    next_time_step = env.step(env_action)

    skill_size = FLAGS.num_skills
    if skill_size > 0:
      cur_observation = time_step.observation[:-skill_size]
      cur_skill = time_step.observation[-skill_size:]
      next_observation = next_time_step.observation[:-skill_size]
    else:
      cur_observation = time_step.observation
      next_observation = next_time_step.observation

    if dynamics is not None:
      if FLAGS.reduced_observation_size:
        process_observation = env.to_dynamics_obs
        cur_observation, next_observation = process_observation(cur_observation), process_observation(next_observation)
      logp = dynamics.get_log_prob(
          np.expand_dims(cur_observation, 0), np.expand_dims(cur_skill, 0),
          np.expand_dims(next_observation, 0))

      cur_predicted_state = np.expand_dims(cur_observation, 0)
      skill_expanded = np.expand_dims(cur_skill, 0)
      cur_predicted_trajectory = [cur_predicted_state[0]]
      for _ in range(predict_trajectory_steps):
        next_predicted_state = dynamics.predict_state(cur_predicted_state,
                                                      skill_expanded)
        cur_predicted_trajectory.append(next_predicted_state[0])
        cur_predicted_state = next_predicted_state
    else:
      logp = ()
      cur_predicted_trajectory = []

    if return_data:
      data.append([
          cur_observation, action_step.action, logp, next_time_step.reward,
          np.array(cur_predicted_trajectory)
      ])
    else:
      extrinsic_reward.append([next_time_step.reward])

    time_step = next_time_step

  if close_environment:
    env.close()

  if return_data:
    return data
  else:
    return extrinsic_reward


def eval_loop(eval_dir,
              eval_policy,
              dynamics=None,
              vid_name=None,
              plot_name=None):
  metadata = tf.io.gfile.GFile(
      os.path.join(eval_dir, 'metadata.txt'), 'a')
  if FLAGS.num_skills == 0:
    num_evals = FLAGS.num_evals
  elif FLAGS.deterministic_eval:
    num_evals = FLAGS.num_skills
  else:
    num_evals = FLAGS.num_evals

  if plot_name is not None:
    palette = sns.color_palette('hls', num_evals)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

  for skill_idx in range(num_evals):
    if FLAGS.num_skills > 0:
      if FLAGS.deterministic_eval:
        preset_skill = np.zeros(FLAGS.num_skills, dtype=np.int64)
        preset_skill[skill_idx] = 1
      elif FLAGS.skill_type == 'discrete_uniform':
        preset_skill = np.random.multinomial(1, [1. / FLAGS.num_skills] *
                                             FLAGS.num_skills)
      elif FLAGS.skill_type == 'gaussian':
        preset_skill = np.random.multivariate_normal(
            np.zeros(FLAGS.num_skills), np.eye(FLAGS.num_skills))
      elif FLAGS.skill_type == 'cont_uniform':
        preset_skill = np.random.uniform(
            low=-1.0, high=1.0, size=FLAGS.num_skills)
      elif FLAGS.skill_type == 'multivariate_bernoulli':
        preset_skill = np.random.binomial(1, 0.5, size=FLAGS.num_skills)
    else:
      preset_skill = None

    eval_env = get_environment(env_name=FLAGS.environment)
    eval_env = wrap_env(
        skill_wrapper.SkillWrapper(
            eval_env,
            num_latent_skills=FLAGS.num_skills,
            skill_type=FLAGS.skill_type,
            preset_skill=preset_skill,
            min_steps_before_resample=FLAGS.min_steps_before_resample,
            resample_prob=FLAGS.resample_prob),
        max_episode_steps=FLAGS.max_env_steps_eval)

    per_skill_evaluations = FLAGS.per_skill_evals
    predict_trajectory_steps = 0

    for eval_idx in range(per_skill_evaluations):

      # record videos for sampled trajectories
      do_record_video = vid_name is not None and eval_idx == 0
      if do_record_video:
        full_vid_name = vid_name + '_' + str(skill_idx)
        traj_env = video_wrapper.VideoWrapper(eval_env, base_path=eval_dir, base_name=full_vid_name)
      else:
        traj_env = eval_env

      eval_trajectory = run_on_env(
          traj_env,
          eval_policy,
          dynamics=dynamics if predict_trajectory_steps > 0 else None,
          predict_trajectory_steps=predict_trajectory_steps,
          return_data=True,
          close_environment=do_record_video)

      trajectory_coordinates = np.asarray([step[0][:2] for step in eval_trajectory])

      if plot_name is not None:
        ax.plot(
            *trajectory_coordinates.T,
            label=(str(skill_idx) if eval_idx == 0 else None),
            c=palette[skill_idx]
        )

        if predict_trajectory_steps > 0:
          for step_idx in range(len(eval_trajectory)):
            if step_idx % 20 == 0:
              ax.plot(eval_trajectory[step_idx][-1][:, 0],
                       eval_trajectory[step_idx][-1][:, 1], 'k:')

      metadata.write(
          str(skill_idx) + ' ' + str(preset_skill) + ' ' +
          str(trajectory_coordinates[-1, :]) + '\n')

  if plot_name is not None:
    fig.tight_layout()
    ax.legend()

    full_image_name = plot_name + '.png'
    # to save images while writing to CNS
    buf = io.BytesIO()
    fig.savefig(buf, dpi=160, bbox_inches='tight')
    buf.seek(0)
    image = tf.io.gfile.GFile(os.path.join(eval_dir, full_image_name), 'w')
    image.write(buf.read(-1))

    # clear before next plot
    plt.clf()


# discrete primitives only, useful with skill-dynamics
def eval_planning(env,
                  dynamics,
                  policy,
                  latent_action_space_size,
                  episode_horizon,
                  planning_horizon=1,
                  primitive_horizon=10,
                  **kwargs):
  """env: tf-agents environment without the skill wrapper."""
  # assuming only discrete action spaces
  high_level_action_space = np.eye(latent_action_space_size)
  time_step = env.reset()

  actual_reward = 0.
  predicted_coords = []

  # planning loop
  for _ in range(episode_horizon // primitive_horizon):
    running_reward = np.zeros(latent_action_space_size)
    running_cur_state = np.array([env.to_dynamics_obs(time_step.observation)] *
                                 latent_action_space_size)
    cur_coord_predicted = [np.expand_dims(running_cur_state[:, :2], 1)]

    # simulate all high level actions for K steps
    for _ in range(planning_horizon):
      predicted_next_state = dynamics.predict_state(running_cur_state,
                                                    high_level_action_space)
      cur_coord_predicted.append(np.expand_dims(predicted_next_state[:, :2], 1))

      # update running stuff
      running_reward += env.compute_reward(running_cur_state, predicted_next_state, info=DADSEnv.OBS_TYPE.FULL_OBS)
      running_cur_state = predicted_next_state

    predicted_coords.append(np.concatenate(cur_coord_predicted, axis=1))

    selected_high_level_action = np.argmax(running_reward)
    for _ in range(primitive_horizon):
      # concatenated observation
      skill_concat_observation = np.concatenate([
          time_step.observation,
          high_level_action_space[selected_high_level_action]
      ],
                                                axis=0)
      next_time_step = env.step(
          np.clip(
              policy.action(
                  hide_coords(
                      time_step._replace(
                          observation=skill_concat_observation))).action,
              -FLAGS.action_clipping, FLAGS.action_clipping))
      actual_reward += next_time_step.reward

      # prepare for next iteration
      time_step = next_time_step

  return actual_reward, predicted_coords


def clip_action(action):
    return clip(action, -FLAGS.action_clipping, FLAGS.action_clipping)


class UniformResampler:
    def __init__(self, env: DADSEnv, buffer: SimpleBuffer, tf_graph: tf.Graph):
        self._buffer = buffer
        self._env = env
        with tf_graph.as_default():
            self._estimator = DensityEstimator(
                input_dim=env.dyn_obs_dim(),
                vae_training_batch_size=FLAGS.agent_batch_size,
                samples_generator=lambda n: self._get_dyn_obs_deltas(buffer.sample(n=n))
            )
        self.tf_session = None

    def _get_dyn_obs_deltas(self, trajectory: Trajectory) -> np.ndarray:
        dyn_obs = self._env.to_dynamics_obs(trajectory.observation[:, 0, :-FLAGS.num_skills])
        next_dyn_obs = self._env.to_dynamics_obs(trajectory.observation[:, 1, :-FLAGS.num_skills])
        return next_dyn_obs - dyn_obs

    def resample(self, num_batches: int, batch_size: int,
                 from_trajectory: Trajectory = None, tb_log_fn: Callable = None) -> List[Trajectory]:
        if from_trajectory is None:
            from_trajectory = self._buffer.sample(num_batches*batch_size)
        deltas = self._get_dyn_obs_deltas(from_trajectory)

        probs = self.tf_session.run(self._estimator.get_input_density(x=deltas))
        tb_log_fn(name="pre-resampling/min(p)", scalar=np.min(probs))
        tb_log_fn(name="pre-resampling/max(p)", scalar=np.max(probs))
        tb_log_fn(name="pre-resampling/mean(p)", scalar=np.mean(probs))

        importance_sampling_weights = 1 / probs.astype(np.float64)
        resampling_probs = importance_sampling_weights / sum(importance_sampling_weights)
        tb_log_fn(name="post-resampling/min(p)", scalar=np.min(resampling_probs))
        tb_log_fn(name="post-resampling/max(p)", scalar=np.max(resampling_probs))
        tb_log_fn(name="post-resampling/mean(p)", scalar=np.mean(resampling_probs))

        all_indices = np.random.choice(len(deltas), size=(num_batches, batch_size), p=resampling_probs)
        return [self.filter_trajectory(from_trajectory, indices) for indices in all_indices]

    def train_density(self) -> None:
        deltas = self._get_dyn_obs_deltas(self._buffer.sample(FLAGS.replay_buffer_capacity))
        self._estimator.VAE.fit(x=deltas, y=deltas, verbose=0, batch_size=256)

    @staticmethod
    def filter_trajectory(trajectory: Trajectory, indices: np.ndarray) -> Trajectory:
        return tf.nest.map_structure(lambda trj: trj[indices], trajectory)


def enter_manual_control_mode(eval_policy: py_tf_policy.PyTFPolicy):
    generator = evaluate_skill_provider_loop(
        env=get_environment(env_name=FLAGS.environment + "_goal"),
        policy=eval_policy,
        episode_length=100,
        hide_coords_fn=hide_coords,
        clip_action_fn=clip_action,
        skill_provider=SliderSkillProvider(num_sliders=FLAGS.num_skills),
        render_env=True
    )
    consume(generator)


def enter_no_control_mode(dynamics: SkillDynamics):
    env: DADSEnv = get_environment(env_name=FLAGS.environment + "_goal")
    generator = evaluate_skill_provider_loop(
        env=env,
        policy=NullActionPolicy(action_dim=env.action_space.shape[0]),
        episode_length=FLAGS.max_env_steps_eval,
        hide_coords_fn=hide_coords,
        clip_action_fn=clip_action,
        skill_provider=NullSkillProvider(skill_dim=FLAGS.num_skills),
        render_env=True
    )
    evaluate_l2_errors(generator, env=env, dynamics=dynamics)


def enter_random_skill_control_mode(policy: py_tf_policy):
    generator = evaluate_skill_provider_loop(
        env=get_environment(env_name=FLAGS.environment + "_goal"),
        policy=policy,
        episode_length=FLAGS.max_env_steps_eval,
        hide_coords_fn=hide_coords,
        clip_action_fn=clip_action,
        skill_provider=RandomSkillProvider(skill_dim=FLAGS.num_skills),
        render_env=True
    )
    consume(generator)


class Policy:
    def action_mean(self, ts: TimeStep) -> np.ndarray:
        raise NotImplementedError


class NullActionPolicy(Policy):
    def __init__(self, action_dim: int):
        self._action_dim = action_dim

    def action_mean(self, ts: TimeStep) -> np.ndarray:
        return np.zeros(self._action_dim)


def calc_dynamics_l2(env: DADSEnv, dynamics: SkillDynamics, dads_steps: Sequence[DADSStep]) -> float:
    cur_obs = np.asarray([s.ts.observation for s in dads_steps])
    skills = np.asarray([s.skill for s in dads_steps])

    next_dyn_obs = env.to_dynamics_obs(np.asarray([s.ts_p1.observation for s in dads_steps]))
    pred_next_obs = dynamics.predict_state(timesteps=env.to_dynamics_obs(cur_obs), actions=skills)

    return l2(next_dyn_obs, pred_next_obs)


def calc_goal_l2(env: DADSEnv, steps: Sequence[DADSStep]) -> float:
    desired_goals = np.asarray([s.goal for s in steps])
    achieved_goals = env.achieved_goal_from_state(np.asarray([s.ts_p1.observation for s in steps]))
    return l2(achieved_goals, desired_goals)


def pct_of_goal_controlling_transitions(env: DADSEnv, trajs: Sequence[Trajectory]) -> float:
    cur_obs = np.vstack([t.observation[:, 0, :] for t in trajs])
    cur_goal = env.achieved_goal_from_state(cur_obs)

    next_obs = np.vstack([t.observation[:, 1, :] for t in trajs])
    next_goal = env.achieved_goal_from_state(next_obs)

    non_moving = _fetch_have_goals_nonmoving(next_goal=next_goal, cur_goal=cur_goal)
    return 1 - non_moving.mean()


def _fetch_have_goals_nonmoving(next_goal: np.ndarray, cur_goal: np.ndarray) -> np.ndarray:
    goal_deltas = np.linalg.norm(next_goal - cur_goal, axis=1)
    fetch_goal_space_diagonal = 0.5
    return goal_deltas < (fetch_goal_space_diagonal/1000)


def main(_):
  logging.info(f"########### USE RESAMPLING SCHEME: {FLAGS.use_dynamics_uniform_resampling}")
  logging.info(f"########### USE STATE SPACE REDUCTION: {FLAGS.use_state_space_reduction}")
  # setting up
  tf.compat.v1.enable_resource_variables()
  tf.compat.v1.disable_eager_execution()
  logging.set_verbosity(logging.INFO)
  global observation_omit_size, sample_count, iter_count, episode_size_buffer, episode_return_buffer

  root_dir = os.path.abspath(os.path.expanduser(FLAGS.logdir))
  if not tf.io.gfile.exists(root_dir):
    tf.io.gfile.makedirs(root_dir)
  experiment_name = f"{FLAGS.experiment_name}-" if FLAGS.experiment_name is not "" else ""
  log_dir = os.path.join(root_dir, FLAGS.environment, f"{experiment_name}seed-{FLAGS.seed}")
  
  if not tf.io.gfile.exists(log_dir):
    tf.io.gfile.makedirs(log_dir)
  save_dir = os.path.join(log_dir, 'models')
  if not tf.io.gfile.exists(save_dir):
    tf.io.gfile.makedirs(save_dir)

  print('directory for recording experiment data:', log_dir)

  # in case training is paused and resumed, so can be restored
  try:
    sample_count = np.load(os.path.join(log_dir, 'sample_count.npy')).tolist()
    iter_count = np.load(os.path.join(log_dir, 'iter_count.npy')).tolist()
    episode_size_buffer = np.load(os.path.join(log_dir, 'episode_size_buffer.npy')).tolist()
    episode_return_buffer = np.load(os.path.join(log_dir, 'episode_return_buffer.npy')).tolist()
  except:
    sample_count = 0
    iter_count = 0
    episode_size_buffer = []
    episode_return_buffer = []

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      os.path.join(log_dir, 'train', 'in_graph_data'), flush_millis=10 * 1000)
  train_summary_writer.set_as_default()

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(True):
    # environment related stuff
    py_env = get_environment(env_name=FLAGS.environment)
    check_reward_fn(env=py_env)
    py_env = wrap_env(
        skill_wrapper.SkillWrapper(
            py_env,
            num_latent_skills=FLAGS.num_skills,
            skill_type=FLAGS.skill_type,
            preset_skill=None,
            min_steps_before_resample=FLAGS.min_steps_before_resample,
            resample_prob=FLAGS.resample_prob),
        max_episode_steps=FLAGS.max_env_steps)

    # all specifications required for all networks and agents
    py_action_spec = py_env.action_spec()
    tf_action_spec = tensor_spec.from_spec(py_action_spec)  # policy, critic action spec
    env_obs_spec = py_env.observation_spec()
    py_env_time_step_spec = ts.time_step_spec(env_obs_spec)  # replay buffer time_step spec
    if observation_omit_size > 0:
      agent_obs_spec = array_spec.BoundedArraySpec(
          (env_obs_spec.shape[0] - observation_omit_size,),
          env_obs_spec.dtype,
          minimum=env_obs_spec.minimum,
          maximum=env_obs_spec.maximum,
          name=env_obs_spec.name)  # policy, critic observation spec
    else:
      agent_obs_spec = env_obs_spec
    py_agent_time_step_spec = ts.time_step_spec(agent_obs_spec)  # policy, critic time_step spec
    tf_agent_time_step_spec = tensor_spec.from_spec(py_agent_time_step_spec)

    # TODO(architsh): Shift co-ordinate hiding to actor_net and critic_net (good for futher image based processing as well)
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_agent_time_step_spec.observation,
        tf_action_spec,
        fc_layer_params=(FLAGS.hidden_layer_size,) * 2,
        continuous_projection_net=_normal_projection_net)

    critic_net = critic_network.CriticNetwork(
        (tf_agent_time_step_spec.observation, tf_action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=(FLAGS.hidden_layer_size,) * 2)

    if FLAGS.skill_dynamics_relabel_type is not None and 'importance_sampling' in FLAGS.skill_dynamics_relabel_type and FLAGS.is_clip_eps > 1.0:
      reweigh_batches_flag = True
    else:
      reweigh_batches_flag = False

    agent = dads_agent.DADSAgent(
        # DADS parameters
        save_dir,
        skill_dynamics_observation_size=py_env.dyn_obs_dim(),
        observation_modify_fn=py_env.to_dynamics_obs,
        restrict_input_size=observation_omit_size,
        latent_size=FLAGS.num_skills,
        latent_prior=FLAGS.skill_type,
        prior_samples=FLAGS.random_skills,
        fc_layer_params=(FLAGS.hidden_layer_size,) * 2,
        normalize_observations=FLAGS.normalize_data,
        network_type=FLAGS.graph_type,
        num_mixture_components=FLAGS.num_components,
        fix_variance=FLAGS.fix_variance,
        reweigh_batches=reweigh_batches_flag,
        skill_dynamics_learning_rate=FLAGS.skill_dynamics_lr,
        # SAC parameters
        time_step_spec=tf_agent_time_step_spec,
        action_spec=tf_action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        target_update_tau=0.005,
        target_update_period=1,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.agent_lr),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.agent_lr),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.agent_lr),
        td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
        gamma=FLAGS.agent_gamma,
        reward_scale_factor=1. /
        (FLAGS.agent_entropy + 1e-12),
        gradient_clipping=None,
        debug_summaries=FLAGS.debug,
        train_step_counter=global_step)

    # evaluation policy
    eval_policy = py_tf_policy.PyTFPolicy(agent.policy)

    # collection policy
    if FLAGS.collect_policy == 'default':
      collect_policy = py_tf_policy.PyTFPolicy(agent.collect_policy)
    elif FLAGS.collect_policy == 'ou_noise':
      collect_policy = py_tf_policy.PyTFPolicy(
          ou_noise_policy.OUNoisePolicy(
              agent.collect_policy, ou_stddev=0.2, ou_damping=0.15))

    # relabelling policy deals with batches of data, unlike collect and eval
    relabel_policy = py_tf_policy.PyTFPolicy(agent.collect_policy)

    # constructing a replay buffer, need a python spec
    policy_step_spec = policy_step.PolicyStep(action=py_action_spec, state=(), info=())

    if FLAGS.skill_dynamics_relabel_type is not None and 'importance_sampling' in FLAGS.skill_dynamics_relabel_type and FLAGS.is_clip_eps > 1.0:
      policy_step_spec = policy_step_spec._replace(
          info=policy_step.set_log_probability(
              policy_step_spec.info,
              array_spec.ArraySpec(
                  shape=(), dtype=np.float32, name='action_log_prob')))

    buffer = SimpleBuffer(capacity=FLAGS.replay_buffer_capacity)

    uniform_resampler = UniformResampler(env=py_env, buffer=buffer, tf_graph=agent._graph)

    # insert experience manually with relabelled rewards and skills
    agent.build_agent_graph()
    agent.build_skill_dynamics_graph()
    agent.create_savers()

    # saving this way requires the saver to be out the object
    train_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(save_dir, 'agent'),
        agent=agent,
        global_step=global_step)
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(save_dir, 'policy'),
        policy=agent.policy,
        global_step=global_step)

    gt.stamp('setup', quick_print=True)

    with tf.compat.v1.Session().as_default() as sess:
      train_checkpointer.initialize_or_restore(sess)
      replay_buffer_saving_fpath = os.path.join(save_dir, 'replay_buffer')
      buffer.load(fpath=replay_buffer_saving_fpath)
      agent.set_sessions(
          initialize_or_restore_skill_dynamics=True, session=sess)
      uniform_resampler.tf_session = sess

      if FLAGS.run_train:

        train_writer = tf.compat.v1.summary.FileWriter(
            os.path.join(log_dir, 'train'), sess.graph)
        common.initialize_uninitialized_variables(sess)
        sess.run(train_summary_writer.init())

        time_step = py_env.reset()
        episode_size_buffer.append(0)
        episode_return_buffer.append(0.)

        # maintain a buffer of episode lengths
        def _process_episodic_data(ep_buffer, cur_data):
          ep_buffer[-1] += cur_data[0]
          ep_buffer += cur_data[1:]

          # only keep the last 100 episodes
          if len(ep_buffer) > 101:
            ep_buffer = ep_buffer[-101:]

        # remove invalid transitions from the replay buffer
        def _filter_trajectories(trajectory):
          # two consecutive samples in the buffer might not have been consecutive in the episode
          valid_indices = (trajectory.step_type[:, 0] != 2)

          return nest.map_structure(lambda x: x[valid_indices], trajectory)

        if FLAGS.manual_control_mode:
            return enter_manual_control_mode(eval_policy=eval_policy)
        if FLAGS.no_control_mode:
            return enter_no_control_mode(dynamics=agent.skill_dynamics)
        if FLAGS.random_skills_control_mode:
            return enter_random_skill_control_mode(policy=eval_policy)

        if iter_count == 0:
          time_step, collect_info = collect_experience(
              py_env,
              time_step,
              collect_policy,
              num_steps=FLAGS.initial_collect_steps,
              replay_buffer=buffer
          )
          _process_episodic_data(episode_size_buffer,
                                 collect_info['episode_sizes'])
          _process_episodic_data(episode_return_buffer,
                                 collect_info['episode_return'])
          sample_count += FLAGS.initial_collect_steps
          gt.stamp('init-collect', quick_print=True)

        for iter_count in gt.timed_for(range(iter_count, FLAGS.num_epochs), name="main-loop"):
          def tb_log(name: str, scalar):
              tensorboard_log_scalar(name=name, scalar=scalar, tb_writer=train_writer, step_num=iter_count)

          fprint('iteration index:', iter_count)

          # model save
          if FLAGS.save_model is not None and iter_count % FLAGS.save_freq == 0:
            fprint('Saving stuff')
            train_checkpointer.save(global_step=iter_count)
            policy_checkpointer.save(global_step=iter_count)
            buffer.save(fpath=replay_buffer_saving_fpath)
            agent.save_variables(global_step=iter_count)

            np.save(os.path.join(log_dir, 'sample_count'), sample_count)
            np.save(os.path.join(log_dir, 'episode_size_buffer'), episode_size_buffer)
            np.save(os.path.join(log_dir, 'episode_return_buffer'), episode_return_buffer)
            np.save(os.path.join(log_dir, 'iter_count'), iter_count)
            gt.stamp("save")

          time_step, collect_info = collect_experience(
              py_env,
              time_step,
              collect_policy,
              num_steps=FLAGS.collect_steps,
              replay_buffer=buffer
          )
          sample_count += FLAGS.collect_steps
          _process_episodic_data(episode_size_buffer,
                                 collect_info['episode_sizes'])
          _process_episodic_data(episode_return_buffer,
                                 collect_info['episode_return'])
          gt.stamp('collect-exp')

          # only for debugging skill relabelling
          if iter_count >= 1 and FLAGS.debug_skill_relabelling:
            trajectory_sample = buffer.sample(5)
            trajectory_sample, is_weights = relabel_skill(
                trajectory_sample,
                relabel_type='importance_sampling',
                cur_policy=relabel_policy,
                cur_skill_dynamics=agent.skill_dynamics,
                env=py_env)
            print(is_weights)

          # DENSITY TRAINING
          if FLAGS.use_dynamics_uniform_resampling:
              uniform_resampler.train_density()
              gt.stamp("density train time")

          def resample_trajs(num_batches: int, batch_size: int) -> List[Trajectory]:
              large_trajectory = buffer.sample(n=FLAGS.skill_dyn_batch_size*FLAGS.skill_dyn_train_steps)
              traj_list = uniform_resampler.resample(
                  num_batches=num_batches,
                  batch_size=batch_size,
                  from_trajectory=large_trajectory,
                  tb_log_fn=tb_log
              )
              return traj_list

          # DYNAMICS TRAINING
          if FLAGS.clear_buffer_every_iter:
              raise NotImplementedError
          elif FLAGS.use_dynamics_uniform_resampling:
              trajectories_list = resample_trajs(num_batches=FLAGS.skill_dyn_train_steps, batch_size=FLAGS.skill_dyn_batch_size)
              gt.stamp("dynamics (re)sampling time")
          else:
              def get_batch():
                  return buffer.sample(n=FLAGS.skill_dyn_batch_size)
              trajectories_list = [get_batch() for _ in range(FLAGS.skill_dyn_train_steps)]
              gt.stamp("[dyn]sample")
          pct = pct_of_goal_controlling_transitions(env=py_env, trajs=trajectories_list)
          tb_log(name="dads/dyn-train-goal-changing-transitions[%]", scalar=pct)

          dynamics_l2_error = []
          # TODO(architsh): clear_buffer_every_iter needs to fix these as well
          for trajectory_sample in trajectories_list:
            # is_weights is None usually, unless relabelling involves importance_sampling
            trajectory_sample, is_weights = relabel_skill(
                trajectory_sample,
                relabel_type=FLAGS.skill_dynamics_relabel_type,
                cur_policy=relabel_policy,
                cur_skill_dynamics=agent.skill_dynamics)
            input_obs = py_env.to_dynamics_obs(
                trajectory_sample.observation[:, 0, :-FLAGS.num_skills])
            cur_skill = trajectory_sample.observation[:, 0, -FLAGS.num_skills:]
            target_obs = py_env.to_dynamics_obs(
                trajectory_sample.observation[:, 1, :-FLAGS.num_skills])
            if FLAGS.clear_buffer_every_iter:
              info = agent.skill_dynamics.train(
                  input_obs,
                  cur_skill,
                  target_obs,
                  batch_size=FLAGS.skill_dyn_batch_size,
                  batch_weights=is_weights,
                  num_steps=FLAGS.skill_dyn_train_steps)
            else:
              info = agent.skill_dynamics.train(
                  input_obs,
                  cur_skill,
                  target_obs,
                  batch_size=-1,
                  batch_weights=is_weights,
                  num_steps=1)
            dynamics_l2_error.append(info["l2-error"])

          if FLAGS.train_skill_dynamics_on_policy:
            buffer.clear()

          gt.stamp("[dyn]train")

          running_dads_reward, running_logp, running_logp_altz = [], [], []

          # AGENT TRAINING
          # agent train loop analysis
          for _ in range(FLAGS.agent_train_steps):
            trajectory_sample = buffer.sample(FLAGS.agent_batch_size)
            gt.stamp("[agent]sample", unique=False)
            trajectory_sample = _filter_trajectories(trajectory_sample)
            trajectory_sample, _ = relabel_skill(
                trajectory_sample,
                relabel_type=FLAGS.agent_relabel_type,
                cur_policy=relabel_policy,
                cur_skill_dynamics=agent.skill_dynamics)

            # need to match the assert structure
            if FLAGS.skill_dynamics_relabel_type is not None and 'importance_sampling' in FLAGS.skill_dynamics_relabel_type:
              trajectory_sample = trajectory_sample._replace(policy_info=())

            if not FLAGS.clear_buffer_every_iter:
              dads_reward, info = agent.train_loop(
                  trajectory_sample,
                  recompute_reward=True,  # turn False for normal SAC training
                  batch_size=-1,
                  num_steps=1)
            else:
              dads_reward, info = agent.train_loop(
                  trajectory_sample,
                  recompute_reward=True,  # turn False for normal SAC training
                  batch_size=FLAGS.agent_batch_size,
                  num_steps=FLAGS.agent_train_steps)
            gt.stamp("[agent]train", unique=False)

            if dads_reward is not None:
              running_dads_reward.append(dads_reward)
              running_logp.append(info['logp'])
              running_logp_altz.append(info['logp_altz'])

          def tb_log(name: str, scalar):
              tensorboard_log_scalar(name=name, scalar=scalar, tb_writer=train_writer, step_num=iter_count)

          if len(episode_size_buffer) > 1:
            tb_log(name='episode_size', scalar=np.mean(episode_size_buffer[:-1]))
            tb_log(name='episode_return', scalar=np.mean(episode_return_buffer[:-1]))

            should_log_state_distribution = False and iter_count % 10 == 0
            if should_log_state_distribution:
                def make_hist(dim, dim_data):
                    return tf.summary.histogram(name=f"tracked-dim/{dim}", data=dim_data, step=sample_count)

                with train_summary_writer.as_default():
                    all_histograms = [make_hist(dim, dim_data) for dim, dim_data in enumerate(input_obs.T)]
                    sess.run(all_histograms)

          tb_log(name='dads/reward', scalar=np.mean(np.concatenate(running_dads_reward)))
          tb_log(name='dads/logp', scalar=np.mean(np.concatenate(running_logp)))
          tb_log(name='dads/logp_altz', scalar=np.mean(np.concatenate(running_logp_altz)))
          tb_log(name="dads/dynamics-l2-error", scalar=np.mean(dynamics_l2_error))

          if FLAGS.clear_buffer_every_iter:
            raise NotImplementedError

          do_perform_eval = (FLAGS.record_freq is not None and
                             iter_count % FLAGS.record_freq == 0 and
                             iter_count > 0)
          if do_perform_eval:
            cur_vid_dir = os.path.join(log_dir, 'videos', str(iter_count))
            tf.io.gfile.makedirs(cur_vid_dir)
            eval_loop(
                cur_vid_dir,
                eval_policy,
                dynamics=agent.skill_dynamics,
                vid_name=FLAGS.vid_name,
                plot_name='traj_plot')

          do_perform_mpc_eval = iter_count > 0 and iter_count % 20 == 0
          if do_perform_mpc_eval:
            env = get_environment(env_name=FLAGS.environment + "_goal")
            skill_provider = MPPISkillProvider(env=env, dynamics=agent.skill_dynamics, skills_to_plan=FLAGS.planning_horizon)
            generator = evaluate_skill_provider_loop(
                env=env,
                policy=eval_policy,
                episode_length=FLAGS.max_env_steps_eval,
                hide_coords_fn=hide_coords,
                clip_action_fn=clip_action,
                skill_provider=skill_provider,
                num_episodes=40)
            steps = list(generator)
            dyn_l2_error = calc_dynamics_l2(dynamics=agent.skill_dynamics, dads_steps=steps, env=env)
            tb_log("DADS-MPC/dynamics-l2-error", dyn_l2_error)
            tb_log("DADS-MPC/goal-l2-error", calc_goal_l2(env=env, steps=steps))
            tb_log("DADS-MPC/rewards", np.mean([s.ts_p1.reward for s in steps]))
          gt.stamp("eval")

        py_env.close()
        try:
            fprint(gt.report(include_itrs=False, include_stats=False, format_options=dict(stamp_name_width=30)))
        except ValueError:
            pass

      # final evaluation, if any
      if FLAGS.run_final_eval:
        vid_dir = os.path.join(log_dir, 'videos', 'final_eval')
        if not tf.io.gfile.exists(vid_dir):
          tf.io.gfile.makedirs(vid_dir)
        vid_name = FLAGS.vid_name

        # generic skill evaluation
        if FLAGS.deterministic_eval or FLAGS.num_evals > 0:
          eval_loop(
              vid_dir,
              eval_policy,
              dynamics=agent.skill_dynamics,
              vid_name=vid_name,
              plot_name='traj_plot')

        # for planning the evaluation directory is changed to save directory
        eval_dir = os.path.join(log_dir, 'eval')
        eval_dir = os.path.join(eval_dir, 'mpc_eval')
        if not tf.io.gfile.exists(eval_dir):
          tf.io.gfile.makedirs(eval_dir)

        eval_plan_env = get_environment(env_name=FLAGS.environment + '_goal')
        record_mpc_performance = False
        if record_mpc_performance:
            eval_plan_env = video_wrapper.VideoWrapper(
                env=eval_plan_env,
                base_path=os.path.join(log_dir, "videos", "final_eval"),
                base_name="final-mpc"
            )

        if 'discrete' in FLAGS.skill_type:
            eval_planning(
                env=eval_plan_env,
                dynamics=agent.skill_dynamics,
                policy=eval_policy,
                latent_action_space_size=FLAGS.num_skills,
                episode_horizon=FLAGS.max_env_steps,
                planning_horizon=FLAGS.planning_horizon,
                primitive_horizon=FLAGS.primitive_horizon
            )
        else:
            skill_provider = MPPISkillProvider(env=eval_plan_env, dynamics=agent.skill_dynamics,
                                               skills_to_plan=FLAGS.planning_horizon)
            generator = evaluate_skill_provider_loop(
                env=eval_plan_env,
                policy=eval_policy,
                episode_length=FLAGS.max_env_steps_eval,
                hide_coords_fn=hide_coords,
                clip_action_fn=clip_action,
                skill_provider=skill_provider,
                render_env=True)
            evaluate_l2_errors(generator, env=eval_plan_env, dynamics=agent.skill_dynamics)

        if record_mpc_performance:
            eval_plan_env.close()


def evaluate_l2_errors(generator, env: DADSEnv, dynamics: SkillDynamics):
    do_log = False
    if do_log:
        filename = f"{FLAGS.environment}-perEpisode-l2.csv"
        with open(filename, "w") as file:
            file.write(f"MEAN_DYN_L2_ERROR,MEAN_GOAL_L2_ERROR\n")
    for steps in grouper(n=FLAGS.max_env_steps_eval, iterable=generator):
        dyn_l2_dist = calc_dynamics_l2(env=env, dynamics=dynamics, dads_steps=steps)
        print(f"Dynamics l2 error: {dyn_l2_dist:.3f}")
        goal_l2_dist = calc_goal_l2(env=env, steps=steps)
        print(f"Goal l2 error: {goal_l2_dist:.3f}")
        if do_log:
            with open(filename, "a") as file:
                file.write(f"{dyn_l2_dist},{goal_l2_dist}\n")


fprint = partial(print, flush=True)


def tensorboard_log_scalar(tb_writer, scalar: float, name: str, step_num: int) -> None:
    val = tf.compat.v1.Summary.Value(tag=name, simple_value=scalar)
    summary = tf.compat.v1.Summary(value=[val])
    tb_writer.add_summary(summary, step_num)


class MPCSkillProvider(SkillProvider):
    def __init__(self, dynamics, env: DADSEnv):
        self._dynamics = dynamics
        self._loop: Generator = None
        self._env = env

    def start_episode(self):
        self._loop = choose_next_skill_loop(dynamics=self._dynamics, env=self._env)

    def get_skill(self, ts: TimeStep):
        next(self._loop)
        return self._loop.send(ts)


def choose_next_skill_loop(dynamics, env: DADSEnv):
    return mppi_next_skill_loop(
        dynamics=dynamics,
        env=env,
        prior_type=FLAGS.prior_type,
        skill_dim=FLAGS.num_skills,
        num_skills_to_plan=FLAGS.planning_horizon,
        steps_per_skill=FLAGS.primitive_horizon,
        refine_steps=FLAGS.refine_steps,
        num_candidate_sequences=FLAGS.num_candidate_sequences,
        smoothing_beta=FLAGS.smoothing_beta,
        mppi_gamma=FLAGS.mppi_gamma,
    )


if __name__ == '__main__':
  tf.compat.v1.app.run(main)
