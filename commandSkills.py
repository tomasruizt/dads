import logging
import os
import pickle
import warnings
from typing import Callable, NamedTuple

import gym
from gym import Wrapper, GoalEnv
from gym.wrappers import FlattenObservation, TimeLimit
from runstats import Statistics
from stable_baselines3 import SAC
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from solvability import ForHER

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

from envs.custom_envs import DADSEnv, make_point2d_dads_env, make_fetch_reach_env, \
    make_fetch_push_env, make_point_mass_env
from scipy.stats import multivariate_normal as mvn


def l2(target, sources: np.ndarray):
    return np.linalg.norm(np.subtract(target, sources), axis=sources.ndim - 1)


class MutualInfoStrategy:
    def __init__(self, skill_dim: int):
        self._skill_dim = skill_dim

    def sample_skill(self, samples=None):
        size = (samples, self._skill_dim) if samples else self._skill_dim
        return np.random.normal(size=size)

    def get_mutual_info(self, goal_delta: np.ndarray, skill: np.ndarray) -> float:
        log = dict()
        log["p(g'|z,g)"] = self._mi_numerator(delta=goal_delta, skill=skill)
        rand_skills = self.sample_skill(1000)
        log["p(g'|g)"] = self._mi_denominator(delta=goal_delta, skills=rand_skills)
        mutual_info = log["p(g'|z,g)"] - log["p(g'|g)"]
        if mutual_info < -0.5:
            logging.warning(str((mutual_info, log["p(g'|z,g)"], log["p(g'|g)"])))
        return mutual_info

    def choose_skill(self, desired_delta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _mi_numerator(delta: np.ndarray, skill: np.ndarray) -> float:
        raise NotImplementedError

    @staticmethod
    def _mi_denominator(delta: np.ndarray, skills: np.ndarray) -> float:
        raise NotImplementedError


class DotProductStrategy(MutualInfoStrategy):
    def sample_skill(self, samples=None):
        skills = super().sample_skill(samples)
        return skills / np.linalg.norm(skills, axis=skills.ndim - 1)[None].T

    @staticmethod
    def _mi_numerator(delta: np.ndarray, skill: np.ndarray) -> float:
        return skill @ delta - np.linalg.norm(delta)

    @staticmethod
    def _mi_denominator(delta: np.ndarray, skills: np.ndarray) -> float:
        return np.log(np.mean(np.exp(skills @ delta - np.linalg.norm(delta))))

    def choose_skill(self, desired_delta: np.ndarray) -> np.ndarray:
        skills = self.sample_skill(samples=100)
        similarity = skills @ desired_delta
        return skills[similarity.argmax()]


class MVNStrategy(MutualInfoStrategy):
    def choose_skill(self, desired_delta: np.ndarray) -> np.ndarray:
        skills = self.sample_skill(samples=100)
        diffs = l2(desired_delta, skills)
        return skills[diffs.argmin()]

    @staticmethod
    def _mi_numerator(delta: np.ndarray, skill: np.ndarray) -> float:
        return mvn.logpdf(x=delta, mean=skill)

    @staticmethod
    def _mi_denominator(delta: np.ndarray, skills: np.ndarray) -> float:
        return np.log(np.mean(mvn.pdf(skills, mean=delta)))


class SkillWrapper(Wrapper):
    def __init__(self, env: DADSEnv, skill_reset_steps: int, first_n_goal_dims=None):
        super().__init__(env)
        self._skill_reset_steps = skill_reset_steps
        if first_n_goal_dims:
            self._skill_dim = first_n_goal_dims
        else:
            self._skill_dim = len(self.env.achieved_goal_from_state(self.env.observation_space.sample()))
        obs_dim = self.env.observation_space.shape[0] + self._skill_dim
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim, ))
        self._sim = MVNStrategy(skill_dim=self._skill_dim)
        self._cur_skill = self._sim.sample_skill()
        self._last_flat_obs = None
        self._goal_deltas_stats = [Statistics([1e-6]) for _ in range(self._skill_dim)]

    def _normalize(self, delta):
        μs = [s.mean() for s in self._goal_deltas_stats]
        σs = [s.stddev() for s in self._goal_deltas_stats]
        # σs = [0.5*max((s.maximum() - μ), μ - s.minimum()) for μ, s in zip(μs, self._goal_deltas_stats)]
        return np.asarray([(d-μ)/σ for (d, μ, σ) in zip(delta, μs, σs)])

    def step(self, action):
        flat_obs, _, done, info = self.env.step(action)
        reward = self._reward(flat_obs)
        self._last_flat_obs = flat_obs
        if np.random.random() < 1/self._skill_reset_steps:
            self._cur_skill = self._sim.sample_skill()
        flat_obs_w_skill = self._add_skill(self._last_flat_obs)
        return flat_obs_w_skill, reward, done, info

    def _reward(self, flat_obs: np.ndarray) -> float:
        last_achieved_goal = self.env.achieved_goal_from_state(self._last_flat_obs)
        achieved_goal = self.env.achieved_goal_from_state(flat_obs)
        goal_delta = (achieved_goal - last_achieved_goal)[:self._skill_dim]
        for s, d in zip(self._goal_deltas_stats, goal_delta):
            s.push(d)
        return self._sim.get_mutual_info(goal_delta=self._normalize(goal_delta),
                                         skill=self._cur_skill)

    def reset(self, **kwargs):
        self._cur_skill = self._sim.sample_skill()
        self._last_flat_obs = self.env.reset(**kwargs)
        return self._add_skill(self._last_flat_obs)

    def _add_skill(self, flat_obs: np.ndarray) -> np.ndarray:
        return np.concatenate((flat_obs, self._cur_skill))

    def set_sac(self, sac):
        self._sac = sac

    def predict(self, dict_obs: dict, deterministic=True):
        delta = (dict_obs["desired_goal"] - dict_obs["achieved_goal"])[:self._skill_dim]
        skill = self._sim.choose_skill(desired_delta=self._normalize(delta))
        flat_obs_w_skill = np.concatenate((dict_obs["observation"], skill))
        return self._sac.predict(observation=flat_obs_w_skill, deterministic=deterministic)

    def save(self, fname: str):
        with open(fname + "-stats.pkl", "wb") as file:
            pickle.dump(self._goal_deltas_stats, file)

    def load(self, fname: str):
        with open(fname + "-stats.pkl", "rb") as file:
            self._goal_deltas_stats = pickle.load(file)


def eval_dict_env(dict_env: GoalEnv, model, ep_len: int):
    while True:
        dict_obs = dict_env.reset()
        for _ in range(ep_len):
            dict_env.render("human")
            action, _ = model.predict(dict_obs, deterministic=True)
            dict_obs, *_ = dict_env.step(action)


def for_sac(env, episode_len: int):
    env = as_dict_env(env)
    env = TimeLimit(env, max_episode_steps=episode_len)
    return FlattenObservation(env)


def as_dict_env(env):
    return ForHER(env)


def set_skills(obs: np.ndarray, skills: np.ndarray) -> None:
    idx = skills.shape[1]
    obs[:, -idx:] = skills


class AddExpCallback(BaseCallback):
    def __init__(self, num_added_samples: int, verbose: int = 0):
        super().__init__(verbose)
        self.num_added_samples = num_added_samples

    def _on_step(self) -> bool:
        buffer: ReplayBuffer = self.model.replay_buffer
        can_sample = buffer.size() > 0
        if not can_sample:
            return True

        samples = buffer.sample(self.num_added_samples)
        new_skills = self.training_env.envs[0]._sample_skill(self.num_added_samples)

        obs = samples.observations.cpu().numpy()
        set_skills(obs, new_skills)

        next_obs = samples.next_observations.cpu().numpy()
        set_skills(next_obs, new_skills)

        goal_delta = self.training_env.envs[0].achieved_goal_from_state(next_obs - obs)
        rewards = -l2(target=goal_delta, sources=new_skills)
        buffer.extend(obs, next_obs, samples.actions.cpu().numpy(), rewards, samples.dones.cpu().numpy())
        return True


envs_fns = dict(
    point2d=make_point2d_dads_env,
    reach=make_fetch_reach_env,
    push=make_fetch_push_env,
    pointmass=make_point_mass_env
)


class Conf(NamedTuple):
    ep_len: int
    num_episodes: int
    lr: float = 3e-4
    first_n_goal_dims: int = None


def show(model, env):
    while True:
        d_obs = env.reset()
        for _ in range(conf.ep_len):
            env.render("human")
            action, _ = model.predict(d_obs, deterministic=True)
            d_obs, *_ = env.step(action)


def train(model, conf: Conf, added_trans: int, save_fname: str):
    kwargs = dict()
    if added_trans > 0:
        kwargs["callback"] = AddExpCallback(num_added_samples=added_trans)
    model.learn(total_timesteps=conf.ep_len * conf.num_episodes, **kwargs)
    model.save(save_fname)


CONFS = dict(
    reach=Conf(ep_len=50, num_episodes=50, lr=0.001),
    point2d=Conf(ep_len=30, num_episodes=50, lr=0.001),
    push=Conf(ep_len=50, num_episodes=2000, first_n_goal_dims=2),
    pointmass=Conf(ep_len=50, num_episodes=500)
)

if __name__ == '__main__':
    as_gdads = True
    name = "pointmass"
    num_added_transtiions = 0

    dads_env_fn = envs_fns[name]
    conf: Conf = CONFS[name]

    if as_gdads:
        env = SkillWrapper(TimeLimit(dads_env_fn(), max_episode_steps=conf.ep_len),
                           skill_reset_steps=conf.ep_len // 2,
                           first_n_goal_dims=conf.first_n_goal_dims)
    else:
        env = for_sac(dads_env_fn(), episode_len=conf.ep_len)
    env = Monitor(env)

    filename = f"modelsCommandSkills/{name}-gdads{as_gdads}"
    if os.path.exists(filename + ".zip"):
        sac = SAC.load(filename, env=env)
        if as_gdads:
            env.load(filename)
    else:
        sac = SAC("MlpPolicy", env=env, verbose=1, learning_rate=conf.lr,
                  tensorboard_log=f"{filename}-tb", buffer_size=10000)
        train(model=sac, conf=conf, added_trans=num_added_transtiions, save_fname=filename)
        if as_gdads:
            env.save(filename)

    if as_gdads:
        env.set_sac(sac)
        eval_dict_env(dict_env=as_dict_env(env=dads_env_fn()),
                      model=env,
                      ep_len=conf.ep_len)
    show(model=sac, env=env)
