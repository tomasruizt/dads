import os
from typing import Callable, NamedTuple

import gym
from gym import Wrapper, GoalEnv
from gym.wrappers import FlattenObservation, TimeLimit
from stable_baselines3 import SAC
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback

from solvability import ForHER

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

from envs.custom_envs import DADSEnv, make_point2d_dads_env, make_fetch_reach_env


def sample_skill_point2d(samples=None):
    size = (samples, 2) if samples else 2
    return np.random.uniform(-0.1, 0.1, size=size)


def sample_skill_reach(samples=None):
    size = (samples, 3) if samples else 3
    max_val = 0.035
    return np.random.uniform(-max_val, max_val, size=size)


def l2(target, sources: np.ndarray):
    return np.linalg.norm(np.subtract(target, sources), axis=sources.ndim - 1)


class SkillWrapper(Wrapper):
    def __init__(self, env: DADSEnv, skill_reset_steps: int, skill_sampling_fn: Callable):
        super().__init__(env)
        self._sample_skill = skill_sampling_fn
        self._skill_reset_steps = skill_reset_steps
        skill_dim = len(self.env.achieved_goal_from_state(self.env.observation_space.sample()))
        obs_dim = self.env.observation_space.shape[0] + skill_dim
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim, ))
        self._cur_skill = self._sample_skill()
        self._last_flat_obs = None

    def step(self, action):
        flat_obs, _, done, info = self.env.step(action)
        reward = self._reward(flat_obs)
        self._last_flat_obs = flat_obs
        if np.random.random() < 1/self._skill_reset_steps:
            self._cur_skill = self._sample_skill()
        flat_obs_w_skill = self._add_skill(self._last_flat_obs)
        return flat_obs_w_skill, reward, done, info

    def _reward(self, flat_obs: np.ndarray) -> float:
        last_achieved_goal = self.env.achieved_goal_from_state(self._last_flat_obs)
        achieved_goal = self.env.achieved_goal_from_state(flat_obs)
        goal_delta = achieved_goal - last_achieved_goal
        return -l2(target=goal_delta, sources=self._cur_skill)

    def reset(self, **kwargs):
        self._cur_skill = self._sample_skill()
        self._last_flat_obs = self.env.reset(**kwargs)
        return self._add_skill(self._last_flat_obs)

    def _add_skill(self, flat_obs: np.ndarray) -> np.ndarray:
        return np.concatenate((flat_obs, self._cur_skill))


class GreedySkillPlanner:
    def __init__(self, sac: SAC, skill_sampling_fn: Callable):
        self._sac = sac
        self._sample_skills = skill_sampling_fn

    def predict(self, dict_obs: dict, deterministic=True):
        delta = dict_obs["desired_goal"] - dict_obs["achieved_goal"]
        skills = self._sample_skills(samples=100)
        diffs = l2(target=delta, sources=skills)
        skill = skills[diffs.argmin()]
        flat_obs_w_skill = np.concatenate((dict_obs["observation"], skill))
        return self._sac.predict(observation=flat_obs_w_skill, deterministic=deterministic)


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
    reach=make_fetch_reach_env
)


class Conf(NamedTuple):
    ep_len: int
    num_episodes: int
    sample_skills_fn: Callable
    lr: float = 3e-4


CONFS = dict(
    reach=Conf(ep_len=50, num_episodes=1000, sample_skills_fn=sample_skill_reach, lr=0.001),
    point2d=Conf(ep_len=30, num_episodes=500, sample_skills_fn=sample_skill_point2d, lr=0.001)
)


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


if __name__ == '__main__':
    as_gdads = True
    name = "reach"
    num_added_transtiions = 0

    dads_env_fn = envs_fns[name]
    conf: Conf = CONFS[name]

    if as_gdads:
        env = SkillWrapper(TimeLimit(dads_env_fn(), max_episode_steps=conf.ep_len),
                           skill_reset_steps=conf.ep_len // 2,
                           skill_sampling_fn=conf.sample_skills_fn)
    else:
        env = for_sac(dads_env_fn(), episode_len=conf.ep_len)

    filename = f"modelsCommandSkills/{name}-gdads{as_gdads}"
    if os.path.exists(filename + ".zip"):
        sac = SAC.load(filename, env=env)
    else:
        sac = SAC("MlpPolicy", env=env, verbose=1, learning_rate=conf.lr)
        train(model=sac, conf=conf, added_trans=num_added_transtiions, save_fname=filename)

    if as_gdads:
        eval_dict_env(dict_env=as_dict_env(env=dads_env_fn()),
                      model=GreedySkillPlanner(sac=sac, skill_sampling_fn=conf.sample_skills_fn),
                      ep_len=conf.ep_len)
    show(model=sac, env=env)
