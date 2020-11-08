from functools import partial
from typing import Callable

import numpy as np

from unsupervised_skill_learning.common_funcs import process_observation_given


def mppi_next_skill_loop(dynamics,
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
                         sparsify_rewards=False):
    if prior_type == "uniform" or sparsify_rewards:
        raise NotImplementedError

    def process(obs):
        return process_observation_given(obs, env_name=env_name, reduced_observation=reduced_observation)

    covariance = 1
    planned_skills_means = np.random.uniform(-1, 1, size=(num_skills_to_plan, skill_dim))
    sample_mvn = partial(np.random.multivariate_normal,
                         cov=covariance * np.eye(skill_dim),
                         size=num_candidate_sequences)
    while True:
        cur_timestep = yield
        for _ in range(refine_steps):
            candidate_skills_seqs = np.tanh([sample_mvn(m) for m in planned_skills_means])
            candidate_skills_seqs = smooth(candidate_skills_seqs, beta=smoothing_beta)
            running_cur_state = np.array([process(cur_timestep.observation)] * num_candidate_sequences)
            running_reward = np.zeros(num_candidate_sequences)
            for cur_skills in candidate_skills_seqs:
                for _ in range(steps_per_skill):
                    pred_next_state = dynamics.predict_state(running_cur_state, cur_skills)
                    dense_reward = env_compute_reward_fn(running_cur_state, pred_next_state, info="dads")
                    running_reward += dense_reward
                    running_cur_state = pred_next_state

            planned_skills_means = update_means(candidate_seqs=candidate_skills_seqs,
                                                rewards=running_reward,
                                                mppi_gamma=mppi_gamma)

        chosen_skill_mean = planned_skills_means[0]
        yield chosen_skill_mean.copy()

        random_skill = np.random.uniform(-1, 1, size=(1, skill_dim))
        planned_skills_means = np.vstack((planned_skills_means[1:], random_skill))


def smooth(skill_seqs, beta):
    β = beta
    plan_len = len(skill_seqs)
    for step in range(1, plan_len):
        cur_skill = skill_seqs[step, ...]
        prev_skill = skill_seqs[step - 1, ...]
        skill_seqs[step, ...] = β * prev_skill + (1. - β) * cur_skill
    return skill_seqs


def update_means(candidate_seqs: np.ndarray, rewards: np.ndarray, mppi_gamma: float):
    rewards = np.exp(mppi_gamma * (rewards - np.max(rewards)))
    rewards = rewards / (rewards.sum() + 1e-10)
    return np.average(candidate_seqs, axis=1, weights=rewards)
