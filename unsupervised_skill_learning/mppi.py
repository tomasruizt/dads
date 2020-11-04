import functools
from typing import Callable

import numpy as np

from lib import py_tf_policy
from unsupervised_skill_learning.common_funcs import process_observation_given, \
    hide_coordinates, clip


def eval_mppi(
    env,
    dynamics,
    policy: py_tf_policy.PyTFPolicy,
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
    steps_per_skill = primitive_horizon
    num_skills_to_plan = planning_horizon

    next_skill_generator = choose_next_skill_loop(
            dynamics=dynamics,
            prior_type=prior_type,
            skill_dim=latent_action_space_size,
            episode_len=episode_horizon,
            num_skills_to_plan=num_skills_to_plan,
            steps_per_skill=steps_per_skill,
            refine_steps=refine_steps,
            num_candidate_sequences=num_candidate_sequences,
            smoothing_beta=smoothing_beta,
            env_name=env_name,
            reduced_observation=reduced_observation,
            sparsify_rewards=sparsify_rewards,
            env_compute_reward_fn=env.compute_reward,
            mppi_gamma=mppi_gamma,
            top_primitives=top_primitives)

    def hide_coords(timestep):
        return hide_coordinates(timestep, first_n=observation_omit_size)

    def clip_action(action):
        return clip(action, low=-action_limit, high=action_limit)

    time_step = env.reset()
    for step in range(episode_horizon):
        next(next_skill_generator)
        chosen_skill = next_skill_generator.send(time_step)
        obs_plus_skill = np.concatenate([time_step.observation, chosen_skill], axis=0)
        action = policy.action_mean(hide_coords(time_step._replace(observation=obs_plus_skill)))
        next_time_step = env.step(clip_action(action))
        time_step = next_time_step
        if step % 5 == 0:
            print(f"MPC step {step}")


def choose_next_skill_loop(dynamics,
                           num_skills_to_plan: int,
                           prior_type: str,
                           skill_dim: int,
                           episode_len: int,
                           steps_per_skill: int,
                           refine_steps: int,
                           num_candidate_sequences,
                           smoothing_beta: float,
                           env_name: str,
                           reduced_observation: int,
                           sparsify_rewards: bool,
                           env_compute_reward_fn: Callable,
                           mppi_gamma: float,
                           top_primitives: int):
    skills_distr_params = []
    for _ in range(num_skills_to_plan):
        skills_distr_params.append(get_initial_skill_params(prior_type=prior_type, dim=skill_dim))

    while True:
        cur_timestep = yield
        for _ in range(episode_len // steps_per_skill):
            for _ in range(refine_steps):
                # generate candidates sequences for primitives
                candidate_skills_seqs = []
                for _ in range(num_candidate_sequences):
                    skill_seq = [sample_skills(skills_distr_params[idx], prior_type=prior_type) for idx in range(num_skills_to_plan)]
                    candidate_skills_seqs.append(skill_seq)
                candidate_skills_seqs = smooth(np.asarray(candidate_skills_seqs), beta=smoothing_beta)

                def process(obs):
                    return process_observation_given(obs, env_name=env_name, reduced_observation=reduced_observation)

                running_cur_state = np.array([process(cur_timestep.observation)] * num_candidate_sequences)
                running_reward = np.zeros(num_candidate_sequences)
                for planning_idx in range(num_skills_to_plan):
                    cur_skills = candidate_skills_seqs[:, planning_idx, :]
                    for _ in range(steps_per_skill):
                        predicted_next_state = dynamics.predict_state(running_cur_state, cur_skills)

                        # update running stuff
                        dense_reward = env_compute_reward_fn(running_cur_state, predicted_next_state, info="dads")
                        # modification for sparse_reward
                        if sparsify_rewards:
                            sparse_reward = 5.0 * (dense_reward > -2) + 0.0 * (
                                    dense_reward <= -2)
                            running_reward += sparse_reward
                        else:
                            running_reward += dense_reward

                        running_cur_state = predicted_next_state

                update_parameters(candidate_skills_seqs, running_reward, skills_distr_params,
                                  prior_type=prior_type, mppi_gamma=mppi_gamma, top_primitives=top_primitives)

        chosen_skill = get_mean_skill(skills_distr_params[0], prior_type=prior_type)
        yield chosen_skill
        skills_distr_params.pop(0)
        skills_distr_params.append(get_initial_skill_params(prior_type=prior_type, dim=skill_dim))


def smooth(primitive_sequences, beta):
    β = beta
    for idx in range(1, primitive_sequences.shape[1]):
      primitive_sequences[:, idx, :] = β * primitive_sequences[:, idx - 1, :] + (1. - β) * primitive_sequences[:, idx, :]
    return primitive_sequences


def get_initial_skill_params(prior_type: str, dim: int):
    if prior_type == 'normal':
        prior_mean = functools.partial(
            np.random.multivariate_normal,
            mean=np.zeros(dim),
            cov=np.diag(np.ones(dim)))
        prior_cov = lambda: 1.5 * np.diag(np.ones(dim))
        return [prior_mean(), prior_cov()]

    elif prior_type == 'uniform':
        prior_low = lambda: np.array([-1.] * dim)
        prior_high = lambda: np.array([1.] * dim)
        return [prior_low(), prior_high()]


def sample_skills(params, prior_type: str):
    if prior_type == 'normal':
        sample = np.random.multivariate_normal(*params)
    elif prior_type == 'uniform':
        sample = np.random.uniform(*params)
    return np.clip(sample, -1., 1.)


def update_parameters(candidates, reward, primitive_parameters,
                      prior_type: str, mppi_gamma: float, top_primitives: int):
    # update new primitive means for horizon sequence
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


def get_mean_skill(params, prior_type: str):
    if prior_type == 'normal':
        return params[0]
    elif prior_type == 'uniform':
        return (params[0] + params[1]) / 2
