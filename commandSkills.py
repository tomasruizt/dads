import os
from multiprocessing import Pool
from typing import NamedTuple

from gym.wrappers import TimeLimit, TransformReward
from itertools import product
import torch

from lib_command_skills import SkillWrapper, GDADSEvalWrapper, as_dict_env, \
    eval_inflen_dict_env, flatten_env
from envs.gym_mujoco.custom_wrappers import DropGoalEnvsAbsoluteLocation

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
from stable_baselines3 import SAC
import numpy as np
from stable_baselines3.common.callbacks import EventCallback
from stable_baselines3.common.monitor import Monitor

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

from envs.custom_envs import make_point2d_dads_env, make_fetch_reach_env, \
    make_fetch_push_env, make_point_mass_env, make_ant_dads_env

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
    layer_size: int = 256
    buffer_size: int = 10_000


def show(model, env, conf: Conf):
    while True:
        d_obs = env.reset()
        for t in range(conf.ep_len):
            env.render("human")
            action, _ = model.predict(d_obs, deterministic=True)
            d_obs, reward, done, info = env.step(action)
            print("step:", t, reward, done, info)


def train(model: SAC, conf: Conf, save_fname: str, eval_env):
    model.learn(total_timesteps=conf.ep_len * conf.num_episodes, log_interval=10,
                callback=[eval_cb(env=eval_env, conf=conf)])
    model.save(save_fname)


CONFS = dict(
    point2d=Conf(ep_len=30, num_episodes=50, lr=0.01),
    reach=Conf(ep_len=50, num_episodes=10*50),
    push=Conf(ep_len=50, num_episodes=1000, skill_dim=2),
    pointmass=Conf(ep_len=150, num_episodes=4*500, reward_scaling=1/50, lr=0.001),
    ant=Conf(ep_len=400, num_episodes=2000, reward_scaling=1/50)
)


class EvalCallbackSuccess(EventCallback):
    def __init__(self, eval_env, conf: Conf, log_path: str, eval_freq: int, n_eval_episodes=5):
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
                               eval_freq=10*conf.ep_len, n_eval_episodes=40)


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
    name = "pointmass"
    drop_abs_position = True

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
        sac = SAC.load(filename + ".zip", env=flat_env)
        if as_gdads:
            flat_env.load(filename)
    else:
        sac = SAC("MlpPolicy", env=flat_env, verbose=1, learning_rate=conf.lr,
                  tensorboard_log=filename, buffer_size=conf.buffer_size, gamma=0.995,
                  learning_starts=4*conf.ep_len, policy_kwargs=dict(net_arch=[conf.layer_size] * 2),
                  seed=seed, device="cpu")
        train(model=sac, conf=conf, save_fname=filename, eval_env=eval_env)
        if as_gdads:
            flat_env.save(filename)
    if render:
        show(model=sac, env=eval_env, conf=conf)


def parallel_main(args):
    return main(*args)


if __name__ == '__main__':
    num_seeds = 5
    do_render = False
    args = product([do_render], range(num_seeds))
    if num_seeds == 1:
        main(*args)
    with Pool() as pool:
        pool.map(parallel_main, args)
