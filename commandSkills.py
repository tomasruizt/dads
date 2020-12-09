import logging
import os
import pickle
from typing import NamedTuple

import gym
from gym import Wrapper, GoalEnv
from gym.wrappers import FlattenObservation, TimeLimit, TransformReward, FilterObservation
from runstats import Statistics
import torch

from envs.gym_mujoco.custom_wrappers import DropGoalEnvsAbsoluteLocation

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
from stable_baselines3 import SAC
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from solvability import ForHER

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

from envs.custom_envs import DADSEnv, make_point2d_dads_env, make_fetch_reach_env, \
    make_fetch_push_env, make_point_mass_env, make_ant_dads_env
from scipy.stats import multivariate_normal as mvn


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
        if mutual_info < -10:
            logging.warning(str((mutual_info, log["p(g'|z,g)"], log["p(g'|g)"])))
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
    def __init__(self, env: GoalEnv, skill_reset_steps: int):
        super().__init__(env)
        self._skill_reset_steps = skill_reset_steps
        self._skill_dim = env.observation_space["desired_goal"].shape[0]
        obs_dim = self.env.observation_space["observation"].shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim + self._skill_dim, ))
        self.strategy = MVNStrategy(skill_dim=self._skill_dim)
        self._cur_skill = self.strategy.sample_skill()
        self._last_dict_obs = None
        self._goal_deltas_stats = [Statistics([1e-6]) for _ in range(self._skill_dim)]

    def _normalize(self, delta):
        μs = [s.mean() for s in self._goal_deltas_stats]
        σs = [s.stddev() for s in self._goal_deltas_stats]
        return np.asarray([(d-μ)/σ for (d, μ, σ) in zip(delta, μs, σs)])

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
        return self.strategy.get_mutual_info(goal_delta=self._normalize(goal_delta),
                                             skill=self._cur_skill)

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


class GDADSEvalWrapper:
    def __init__(self, dict_env, sw: SkillWrapper):
        self.dict_env = dict_env
        self._sw = sw

    def render(self, *args, **kwargs):
        return self.dict_env.render(*args, **kwargs)

    def step(self, action):
        dict_obs, *step = self.dict_env.step(action)
        skill = self._sw.best_skill_for(dict_obs)
        flat_obs_w_skill = np.concatenate((dict_obs["observation"], skill))
        return (flat_obs_w_skill, *step)

    def reset(self):
        dict_obs = self.dict_env.reset()
        skill = self._sw.best_skill_for(dict_obs)
        return np.concatenate((dict_obs["observation"], skill))


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
        wrapper: SkillWrapper = self.training_env.envs[0]
        new_samples = wrapper.relabel(**{k:v.cpu().numpy() for k, v in samples._asdict().items()})
        buffer.extend(*new_samples)
        return True


envs_fns = dict(
    point2d=make_point2d_dads_env,
    reach=make_fetch_reach_env,
    push=make_fetch_push_env,
    pointmass=make_point_mass_env,
    ant=make_ant_dads_env
)


class Conf(NamedTuple):
    ep_len: int
    num_episodes: int
    lr: float = 3e-4
    first_n_goal_dims: int = None
    reward_scaling: float = 1.0


def show(model, env, conf: Conf):
    while True:
        d_obs = env.reset()
        for _ in range(conf.ep_len):
            env.render("human")
            action, _ = model.predict(d_obs, deterministic=True)
            d_obs, *_ = env.step(action)


def train(model: SAC, conf: Conf, save_fname: str):
    model.learn(total_timesteps=conf.ep_len * conf.num_episodes, log_interval=10,
                callback=[eval_cb(model.env)])
    model.save(save_fname)


CONFS = dict(
    point2d=Conf(ep_len=30, num_episodes=50, lr=0.01),
    reach=Conf(ep_len=50, num_episodes=50, lr=0.001),
    push=Conf(ep_len=50, num_episodes=2000, first_n_goal_dims=2),
    pointmass=Conf(ep_len=150, num_episodes=300, lr=0.001, reward_scaling=1/100),
    ant=Conf(ep_len=200, num_episodes=5000, reward_scaling=1/50)
)


def eval_cb(env):
    return EvalCallback(eval_env=env, n_eval_episodes=10, log_path="modelsCommandSkills", deterministic=True)


def save_cb(name: str):
    return CheckpointCallback(save_freq=10000, save_path="modelsCommandSkills", name_prefix=name)


def main():
    as_gdads = True
    name = "point2d"
    drop_abs_position = True

    dads_env_fn = envs_fns[name]
    conf: Conf = CONFS[name]

    dict_env = as_dict_env(dads_env_fn())
    dict_env = TimeLimit(dict_env, max_episode_steps=conf.ep_len)
    if drop_abs_position:
        dict_env = DropGoalEnvsAbsoluteLocation(dict_env)
    if as_gdads:
        flat_env = SkillWrapper(env=dict_env, skill_reset_steps=conf.ep_len // 2)
    else:
        flat_obs_content = ["observation", "desired_goal", "achieved_goal"]
        if drop_abs_position:
            flat_obs_content.remove("achieved_goal")  # Because always 0 vector
        flat_env = FlattenObservation(FilterObservation(dict_env, filter_keys=flat_obs_content))

    flat_env = TransformReward(flat_env, f=lambda r: r*conf.reward_scaling)
    flat_env = Monitor(flat_env)

    filename = f"modelsCommandSkills/{name}-gdads{as_gdads}"
    if os.path.exists(filename + ".zip"):
        sac = SAC.load(filename, env=flat_env)
        if as_gdads:
            flat_env.load(filename)
    else:
        sac = SAC("MlpPolicy", env=flat_env, verbose=1, learning_rate=conf.lr,
                  tensorboard_log=f"{filename}-tb", buffer_size=10000)
        train(model=sac, conf=conf, save_fname=filename)
        if as_gdads:
            flat_env.save(filename)

    eval_env = flat_env if not as_gdads else GDADSEvalWrapper(dict_env, sw=flat_env)
    show(model=sac, env=eval_env, conf=conf)


if __name__ == '__main__':
    main()
