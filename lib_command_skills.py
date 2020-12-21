import logging
import pickle
from typing import Generator, Tuple

import gym
import numpy as np
from gym import Wrapper, GoalEnv
from gym.wrappers import FlattenObservation, FilterObservation
from runstats import Statistics
from scipy.stats import multivariate_normal as mvn
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, \
    EveryNTimesteps

from solvability import ForHER


def l2(target, sources: np.ndarray):
    return np.linalg.norm(np.subtract(target, sources), axis=sources.ndim - 1)


class MutualInfoStrategy:
    def __init__(self, skill_dim: int):
        self._skill_dim = skill_dim

    def sample_skill(self, samples=None):
        raise NotImplementedError

    def get_mutual_info(self, goal_delta: np.ndarray, skill: np.ndarray) -> float:
        log = dict()
        log["p(g'|z,g)"] = self._mi_numerator(delta=goal_delta, skill=skill)
        log["p(g'|g)"] = self._mi_denominator(delta=goal_delta)
        mutual_info = log["p(g'|z,g)"] - log["p(g'|g)"]
        if mutual_info < -20:
            logging.warning("Low Mutual Info %s" % str((mutual_info, log["p(g'|z,g)"], log["p(g'|g)"])))
        return mutual_info

    def choose_skill(self, desired_delta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _mi_numerator(self, delta: np.ndarray, skill: np.ndarray) -> float:
        raise NotImplementedError

    def _mi_denominator(self, delta: np.ndarray) -> float:
        raise NotImplementedError


class DotProductStrategy(MutualInfoStrategy):
    def sample_skill(self, samples=None):
        size = (samples, self._skill_dim) if samples else self._skill_dim
        return np.random.normal(size=size)

    def _mi_numerator(self, delta: np.ndarray, skill: np.ndarray) -> float:
        diff = delta - skill
        return -0.5 * (diff @ diff)

    def _mi_denominator(self, delta: np.ndarray) -> float:
        return -0.25*(delta @ delta) - np.log(np.sqrt(2**len(delta)))

    def choose_skill(self, desired_delta: np.ndarray) -> np.ndarray:
        skills = self.sample_skill(samples=1000)
        diffs = l2(desired_delta, skills)
        return skills[diffs.argmin()]


class MVNStrategy(MutualInfoStrategy):
    def __init__(self, skill_dim: int):
        super().__init__(skill_dim)
        self.cov = cov = {
            "z": np.eye(self._skill_dim),
            "g'|z,g": 0.1 * np.eye(self._skill_dim)
        }
        # Integration of two gaussians <-> convolution <-> sum of two gaussian RVs.
        cov["g'|g"] = cov["z"] + cov["g'|z,g"]

    def sample_skill(self, samples=1):
        return mvn.rvs(size=samples, cov=self.cov["z"])

    def choose_skill(self, desired_delta: np.ndarray) -> np.ndarray:
        skills = self.sample_skill(samples=1000)
        diffs = l2(desired_delta, skills)
        return skills[diffs.argmin()]

    def _mi_numerator(self, delta: np.ndarray, skill: np.ndarray) -> float:
        return mvn.logpdf(x=delta, mean=skill, cov=self.cov["g'|z,g"])

    def _mi_denominator(self, delta: np.ndarray) -> float:
        return mvn.logpdf(x=delta, cov=self.cov["g'|g"])


class SkillWrapper(Wrapper):
    def __init__(self, env: GoalEnv, skill_reset_steps: int, skill_dim=None):
        super().__init__(env)
        self._skill_reset_steps = skill_reset_steps
        self._skill_dim = skill_dim or env.observation_space["desired_goal"].shape[0]
        obs_dim = self.env.observation_space["observation"].shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim + self._skill_dim, ))
        self.strategy = MVNStrategy(skill_dim=self._skill_dim)
        self._cur_skill = self.strategy.sample_skill()
        self._last_dict_obs = None
        self._goal_deltas_stats = [Statistics([1e-6]) for _ in range(self._skill_dim)]

    def _normalize(self, delta):
        _, σs, maxs = self.get_deltas_statistics()
        ns = [np.mean((σ, m)) for σ, m in zip(σs, maxs)]
        return np.asarray([d/n for d, n in zip(delta, ns)])

    def get_deltas_statistics(self) -> Tuple:
        μs = [s.mean() for s in self._goal_deltas_stats]
        σs = [s.stddev() for s in self._goal_deltas_stats]
        maxs = [s.maximum() for s in self._goal_deltas_stats]
        return μs, σs, maxs

    def step(self, action):
        dict_obs, _, done, info = self.env.step(action)
        reward = self._reward(dict_obs=dict_obs)
        self._last_dict_obs = dict_obs
        if np.random.random() < 1/self._skill_reset_steps:
            self._cur_skill = self.strategy.sample_skill()
        flat_obs_w_skill = self._add_skill(observation=dict_obs["observation"])
        return flat_obs_w_skill, reward, done, info

    def _reward(self, dict_obs: np.ndarray) -> float:
        last_diff = self._last_dict_obs["achieved_goal"] - self._last_dict_obs["desired_goal"]
        cur_diff = dict_obs["achieved_goal"] - dict_obs["desired_goal"]
        goal_delta = (cur_diff - last_diff)[:self._skill_dim]
        for s, d in zip(self._goal_deltas_stats, goal_delta):
            s.push(d)
        mi = self.strategy.get_mutual_info(goal_delta=self._normalize(goal_delta),
                                           skill=self._cur_skill)
        if mi < -100:
            logging.warning(f"Mutual information very low: {mi}. Clipping it to -100")
        return max(mi, -100)

    def reset(self, **kwargs):
        self._cur_skill = self.strategy.sample_skill()
        self._last_dict_obs = self.env.reset(**kwargs)
        return self._add_skill(observation=self._last_dict_obs["observation"])

    def _add_skill(self, observation: np.ndarray) -> np.ndarray:
        return np.concatenate((observation, self._cur_skill))

    def best_skill_for(self, dict_obs):
        delta = (dict_obs["desired_goal"] - dict_obs["achieved_goal"])[:self._skill_dim]
        return self.strategy.choose_skill(desired_delta=self._normalize(delta))

    def save(self, fname: str):
        with open(fname + "-stats.pkl", "wb") as file:
            pickle.dump(self._goal_deltas_stats, file)

    def load(self, fname: str):
        with open(fname + "-stats.pkl", "rb") as file:
            self._goal_deltas_stats = pickle.load(file)

    def relabel(self, observations, actions, next_observations, rewards, dones):
        assert observations.ndim == 2, observations.ndim
        assert observations.shape == next_observations.shape, (observations.shape, next_observations.shape)
        deltas = self.env.achieved_goal_from_state(next_observations - observations)[:self._skill_dim]
        deltas = self._normalize(deltas)

        new_skills = self.strategy.sample_skill(len(observations))
        mi = self.strategy.get_mutual_info
        rewards = np.asarray([mi(goal_delta=d, skill=s) for d, s in zip(deltas, new_skills)])

        new_obs, new_next_obs = observations.copy(), next_observations.copy()
        set_skills(observations, new_skills)
        set_skills(next_observations, new_skills)
        return new_obs, new_next_obs, actions, rewards, dones


class GDADSEvalWrapper(Wrapper):
    def __init__(self, dict_env, sw: SkillWrapper):
        super().__init__(dict_env)
        self._sw = sw
        self.observation_space = self._sw.observation_space

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def step(self, action):
        dict_obs, *step = self.env.step(action)
        skill = self._sw.best_skill_for(dict_obs)
        flat_obs_w_skill = np.concatenate((dict_obs["observation"], skill))
        return (flat_obs_w_skill, *step)

    def reset(self):
        dict_obs = self.env.reset()
        skill = self._sw.best_skill_for(dict_obs)
        return np.concatenate((dict_obs["observation"], skill))


def as_dict_env(env):
    return ForHER(env)


def set_skills(obs: np.ndarray, skills: np.ndarray) -> None:
    idx = skills.shape[1]
    obs[:, -idx:] = skills


class LogDeltaStatistics(EveryNTimesteps):
    def __init__(self, n_steps: int, callback: BaseCallback = None):
        super().__init__(n_steps, callback)

    def _on_event(self) -> bool:
        μs, σs, maxs = self.training_env.envs[0].get_deltas_statistics()
        self.logger.record("deltas-stats/max(abs(μs))", np.max(np.abs(μs)))
        self.logger.record("deltas-stats/min(σs)", np.min(σs))
        self.logger.record("deltas-stats/max(deltas)", np.max(maxs))
        return True


def eval_inflen_dict_env(model, env, episode_len: int) -> Generator:
        dict_obs = env.reset()
        for _ in range(episode_len):
            action, _ = model.predict(dict_obs, deterministic=True)
            dict_obs, reward, done, info = env.step(action)
            yield reward, info


def save_cb(name: str):
    return CheckpointCallback(save_freq=10000, save_path="modelsCommandSkills", name_prefix=name)


def flatten_env(dict_env, drop_abs_position):
    flat_obs_content = ["observation", "desired_goal", "achieved_goal"]
    if drop_abs_position:
        flat_obs_content.remove("achieved_goal")  # Because always 0 vector
    return FlattenObservation(FilterObservation(dict_env, filter_keys=flat_obs_content))