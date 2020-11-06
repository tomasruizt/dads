import functools
from typing import Callable

import numpy as np

from lib import py_tf_policy
from unsupervised_skill_learning.common_funcs import process_observation_given


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


def choose_next_skill_loop_given(dynamics,
                                 num_skills_to_plan: int,
                                 prior_type: str,
                                 skill_dim: int,
                                 steps_per_skill: int,
                                 refine_steps: int,
                                 num_candidate_sequences,
                                 smoothing_beta: float,
                                 env_name: str,
                                 reduced_observation: int,
                                 env_compute_reward_fn: Callable,
                                 mppi_gamma: float,
                                 top_primitives: int,
                                 sparsify_rewards=False):
    skills_distr_params = []
    for _ in range(num_skills_to_plan):
        skills_distr_params.append(get_initial_skill_params(prior_type=prior_type, dim=skill_dim))

    while True:
        cur_timestep = yield
        for _ in range(refine_steps):
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
                        sparse_reward = 5.0 * (dense_reward > -2) + 0.0 * (dense_reward <= -2)
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
        new_means = np.average(candidates, axis=0, weights=reward)

        for idx, mean in enumerate(new_means):
            primitive_parameters[idx][0] = mean
        return

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
