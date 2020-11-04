import functools

import numpy as np

from unsupervised_skill_learning.common_funcs import process_observation_given, \
    hide_coordinates, clip


def eval_mppi(
    env,
    dynamics,
    policy,
    latent_action_space_size,
    episode_horizon,
    env_name,
    planning_horizon=1,
    primitive_horizon=10,
    num_candidate_sequences=50,
    refine_steps=10,
    mppi_gamma=10,
    prior_type='normal',
    smoothing_beta=0.9,
    # no need to change generally
    sparsify_rewards=False,
    # only for uniform prior mode
    top_primitives=5,
    action_limit=1,
    reduced_observation=0,
    observation_omit_size=0
):
  """env: tf-agents environment without the skill wrapper.
     dynamics: skill-dynamics model learnt by DADS.
     policy: skill-conditioned policy learnt by DADS.
     planning_horizon: number of latent skills to plan in the future.
     primitive_horizon: number of steps each skill is executed for.
     num_candidate_sequences: number of samples executed from the prior per
     refining step of planning.
     refine_steps: number of steps for which the plan is iterated upon before
     execution (number of optimization steps).
     mppi_gamma: MPPI parameter for reweighing rewards.
     prior_type: 'normal' implies MPPI, 'uniform' implies a CEM like algorithm
     (not tested).
     smoothing_beta: for planning_horizon > 1, the every sampled plan is
     smoothed using EMA. (0-> no smoothing, 1-> perfectly smoothed)
     sparsify_rewards: converts a dense reward problem into a sparse reward
     (avoid using).
     top_primitives: number of elites to choose, if using CEM (not tested).
  """

  def smooth(primitive_sequences):
    β = smoothing_beta
    for planning_idx in range(1, primitive_sequences.shape[1]):
      primitive_sequences[:, planning_idx, :] = β * primitive_sequences[:, planning_idx - 1, :] + (1. - β) * primitive_sequences[:, planning_idx, :]

    return primitive_sequences

  def _get_init_primitive_parameters():
    if prior_type == 'normal':
      prior_mean = functools.partial(
          np.random.multivariate_normal,
          mean=np.zeros(latent_action_space_size),
          cov=np.diag(np.ones(latent_action_space_size)))
      prior_cov = lambda: 1.5 * np.diag(np.ones(latent_action_space_size))
      return [prior_mean(), prior_cov()]

    elif prior_type == 'uniform':
      prior_low = lambda: np.array([-1.] * latent_action_space_size)
      prior_high = lambda: np.array([1.] * latent_action_space_size)
      return [prior_low(), prior_high()]

  def _sample_primitives(params):
    if prior_type == 'normal':
      sample = np.random.multivariate_normal(*params)
    elif prior_type == 'uniform':
      sample = np.random.uniform(*params)
    return np.clip(sample, -1., 1.)

  # update new primitive means for horizon sequence
  def _update_parameters(candidates, reward, primitive_parameters):
    # a more regular mppi
    if prior_type == 'normal':
      reward = np.exp(mppi_gamma * (reward - np.max(reward)))
      reward = reward / (reward.sum() + 1e-10)
      new_means = (candidates.T * reward).T.sum(axis=0)

      for planning_idx in range(candidates.shape[1]):
        primitive_parameters[planning_idx][0] = new_means[planning_idx]

    # TODO(architsh): closer to cross-entropy/shooting method, figure out a better update
    elif prior_type == 'uniform':
      chosen_candidates = candidates[np.argsort(reward)[-top_primitives:]]
      candidates_min = np.min(chosen_candidates, axis=0)
      candidates_max = np.max(chosen_candidates, axis=0)

      for planning_idx in range(candidates.shape[1]):
        primitive_parameters[planning_idx][0] = candidates_min[planning_idx]
        primitive_parameters[planning_idx][1] = candidates_max[planning_idx]

  def _get_expected_primitive(params):
    if prior_type == 'normal':
      return params[0]
    elif prior_type == 'uniform':
      return (params[0] + params[1]) / 2

  time_step = env.reset()
  actual_reward = 0.

  primitive_parameters = []
  chosen_primitives = []
  for _ in range(planning_horizon):
    primitive_parameters.append(_get_init_primitive_parameters())

  for _ in range(episode_horizon // primitive_horizon):
    for _ in range(refine_steps):
      # generate candidates sequences for primitives
      candidate_primitive_sequences = []
      for _ in range(num_candidate_sequences):
        candidate_primitive_sequences.append([
            _sample_primitives(primitive_parameters[planning_idx])
            for planning_idx in range(planning_horizon)
        ])

      candidate_primitive_sequences = np.array(candidate_primitive_sequences)
      candidate_primitive_sequences = smooth(
          candidate_primitive_sequences)

      def process_observation(obs):
          return process_observation_given(obs, env_name=env_name, reduced_observation=reduced_observation)

      running_cur_state = np.array(
          [process_observation(time_step.observation)] *
          num_candidate_sequences)
      running_reward = np.zeros(num_candidate_sequences)
      for planning_idx in range(planning_horizon):
        cur_primitives = candidate_primitive_sequences[:, planning_idx, :]
        for _ in range(primitive_horizon):
          predicted_next_state = dynamics.predict_state(running_cur_state,
                                                        cur_primitives)

          # update running stuff
          dense_reward = env.compute_reward(running_cur_state, predicted_next_state, info="dads")
          # modification for sparse_reward
          if sparsify_rewards:
            sparse_reward = 5.0 * (dense_reward > -2) + 0.0 * (
                dense_reward <= -2)
            running_reward += sparse_reward
          else:
            running_reward += dense_reward

          running_cur_state = predicted_next_state

      _update_parameters(candidate_primitive_sequences, running_reward,
                         primitive_parameters)

    chosen_primitive = _get_expected_primitive(primitive_parameters[0])
    chosen_primitives.append(chosen_primitive)

    # a loop just to check what the chosen primitive is expected to do
    # running_cur_state = np.array([process_observation(time_step.observation)])
    # for _ in range(primitive_horizon):
    #   predicted_next_state = dynamics.predict_state(
    #       running_cur_state, np.expand_dims(chosen_primitive, 0))
    #   running_cur_state = predicted_next_state
    # print('Predicted next co-ordinates:', running_cur_state[0, :2])

    def hide_coords(timestep):
        return hide_coordinates(timestep, first_n=observation_omit_size)

    def clip_action(action):
        return clip(action, low=-action_limit, high=action_limit)

    for _ in range(primitive_horizon):
      # concatenated observation
      obs_plus_skill = np.concatenate([time_step.observation, chosen_primitive], axis=0)
      action = policy.action(hide_coords(time_step._replace(observation=obs_plus_skill))).action
      next_time_step = env.step(clip_action(action))
      actual_reward += next_time_step.reward
      # prepare for next iteration
      time_step = next_time_step
      # print(step_idx)

    primitive_parameters.pop(0)
    primitive_parameters.append(_get_init_primitive_parameters())

  return actual_reward, np.array(chosen_primitives)