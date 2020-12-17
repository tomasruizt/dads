import logging
import os
import pickle
from multiprocessing import Pool
from typing import NamedTuple, Generator

import gym
from gym import Wrapper, GoalEnv
from gym.wrappers import FlattenObservation, TimeLimit, TransformReward, FilterObservation
from itertools import product
from runstats import Statistics
import torch

from envs.gym_mujoco.custom_wrappers import DropGoalEnvsAbsoluteLocation

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
from stable_baselines3 import SAC
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EventCallback
from stable_baselines3.common.monitor import Monitor

from solvability import ForHER

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

from envs.custom_envs import make_point2d_dads_env, make_fetch_reach_env, \
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
            "g'|z,g": 1 * np.eye(self._skill_dim)
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
    skill_dim: int = None
    reward_scaling: float = 1.0
    collect_steps = 500
    layer_size = 256


def show(model, env, conf: Conf):
    while True:
        d_obs = env.reset()
        for t in range(conf.ep_len):
            env.render("human")
            action, _ = model.predict(d_obs, deterministic=True)
            d_obs, reward, done, info = env.step(action)
            print("step:", t, reward, done, info)


def eval_inflen_dict_env(model, env, episode_len: int) -> Generator:
        dict_obs = env.reset()
        for _ in range(episode_len):
            action, _ = model.predict(dict_obs, deterministic=True)
            dict_obs, reward, done, info = env.step(action)
            yield reward, info


def train(model: SAC, conf: Conf, save_fname: str, eval_env):
    model.learn(total_timesteps=conf.ep_len * conf.num_episodes, log_interval=10,
                callback=[eval_cb(env=eval_env, conf=conf)])
    model.save(save_fname)


CONFS = dict(
    point2d=Conf(ep_len=30, num_episodes=50, lr=0.01),
    reach=Conf(ep_len=50, num_episodes=10*50),
    push=Conf(ep_len=50, num_episodes=1000, skill_dim=2),
    pointmass=Conf(ep_len=150, num_episodes=300, lr=0.001, reward_scaling=1/100),
    ant=Conf(ep_len=400, num_episodes=2000, reward_scaling=1/50)
)


class EvalCallbackSuccess(EventCallback):
    def __init__(self, eval_env, conf: Conf, log_path: str, eval_freq: int, n_eval_episodes = 5):
        super().__init__(None, verbose=1)
        self._conf = conf
        self._n_eval_episodes = n_eval_episodes
        self._eval_freq = eval_freq
        self._eval_env = eval_env
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path

    def _init_callback(self) -> None:
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:
        do_log = self._eval_freq > 0 and self.n_calls % self._eval_freq == 0
        if do_log:
            self._log()
        return True

    def _log(self):
        rewards = []
        successes = []
        for _ in range(self._n_eval_episodes):
            results = list(eval_inflen_dict_env(model=self.model, env=self._eval_env, episode_len=self._conf.ep_len))
            rewards.append(np.mean([rew for rew, _ in results]))
            successes.append(results[-1][1]["is_success"])
        self.logger.record("DADS-MPC/is-success", np.mean(successes))
        self.logger.record("DADS-MPC/rewards", np.mean(rewards))


def eval_cb(env, conf: Conf):
    return EvalCallbackSuccess(eval_env=env, conf=conf, log_path="modelsCommandSkills",
                               eval_freq=20*conf.ep_len, n_eval_episodes=3)


def save_cb(name: str):
    return CheckpointCallback(save_freq=10000, save_path="modelsCommandSkills", name_prefix=name)


def get_env(name: str, drop_abs_position: bool):
    dads_env_fn = envs_fns[name]
    conf: Conf = CONFS[name]

    dict_env = as_dict_env(dads_env_fn())
    dict_env = TimeLimit(dict_env, max_episode_steps=conf.ep_len)
    if drop_abs_position:
        dict_env = DropGoalEnvsAbsoluteLocation(dict_env)
    return dict_env


def main(render=True, seed=0):
    as_gdads = True
    name = "reach"
    drop_abs_position = False

    conf: Conf = CONFS[name]
    dict_env = get_env(name=name, drop_abs_position=drop_abs_position)
    if as_gdads:
        flat_env = SkillWrapper(env=dict_env, skill_reset_steps=conf.ep_len // 2)
    else:
        flat_env = flatten_env(dict_env, drop_abs_position)
    flat_env = TransformReward(flat_env, f=lambda r: r*conf.reward_scaling)
    flat_env = Monitor(flat_env)

    dict_env = get_env(name=name, drop_abs_position=drop_abs_position)
    if as_gdads:
        eval_env = GDADSEvalWrapper(dict_env, sw=flat_env)
    else:
        eval_env = flatten_env(dict_env=dict_env, drop_abs_position=drop_abs_position)

    filename = f"modelsCommandSkills/{name}/asGDADS{as_gdads}/resamplingFalse_goalSpaceTrue-seed-{seed}"
    if os.path.exists(filename + ".zip"):
        sac = SAC.load(filename, env=flat_env)
        if as_gdads:
            flat_env.load(filename)
    else:
        sac = SAC("MlpPolicy", env=flat_env, verbose=1, learning_rate=conf.lr,
                  tensorboard_log=filename, buffer_size=100000, gamma=0.995,
                  learning_starts=conf.ep_len, policy_kwargs=dict(net_arch=[conf.layer_size] * 2),
                  train_freq=conf.collect_steps, gradient_steps=64, seed=seed, device="cpu")
        train(model=sac, conf=conf, save_fname=filename, eval_env=eval_env)
        if as_gdads:
            flat_env.save(filename)
    if render:
        show(model=sac, env=eval_env, conf=conf)


def flatten_env(dict_env, drop_abs_position):
    flat_obs_content = ["observation", "desired_goal", "achieved_goal"]
    if drop_abs_position:
        flat_obs_content.remove("achieved_goal")  # Because always 0 vector
    return FlattenObservation(FilterObservation(dict_env, filter_keys=flat_obs_content))


def parallel_main(args):
    return main(*args)


if __name__ == '__main__':
    num_seeds = 8
    render = False
    args = product([render], range(num_seeds))
    with Pool() as pool:
        pool.map(parallel_main, args)
