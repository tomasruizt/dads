import os
from collections import OrderedDict
from multiprocessing import Pool
from typing import NamedTuple

import gym
from gym import Wrapper, ObservationWrapper
from gym.wrappers import TimeLimit, TransformReward
from itertools import product
import torch

from lib_command_skills import SkillWrapper, GDADSEvalWrapper, as_dict_env, \
    eval_inflen_dict_env, flatten_env, LogDeltaStatistics, BestSkillProvider, \
    ant_grid_evaluation, dump_ant_grid_evaluation
from envs.gym_mujoco.custom_wrappers import DropGoalEnvsAbsoluteLocation
from skill_slider import create_sliders_widget

torch.set_num_threads(6)
torch.set_num_interop_threads(6)
from stable_baselines3 import SAC
import numpy as np
from stable_baselines3.common.callbacks import EventCallback, EveryNTimesteps, BaseCallback, \
    CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from solvability import TimeFeature

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
    eval_freq_in_episodes: int = 50
    lr: float = 3e-4
    skill_dim: int = None
    reward_scaling: float = 1.0
    layer_size: int = 256
    batch_size: int = 256
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
    model.env.reset()
    model.learn(total_timesteps=conf.ep_len * conf.num_episodes,
                log_interval=10,
                callback=[eval_cb(env=eval_env, conf=conf, save_fname=save_fname),
                          LogDeltaStatistics(n_steps=conf.ep_len),
                          LogDeltasHistogram(env=model.env, freq_in_steps=25 * conf.ep_len)],
                reset_num_timesteps=False)


CONFS = dict(
    point2d=Conf(ep_len=30, num_episodes=400, lr=0.01, eval_freq_in_episodes=25),
    reach=Conf(ep_len=50, num_episodes=10*50),
    push=Conf(ep_len=50, num_episodes=6*1000, skill_dim=2, layer_size=512, buffer_size=100_000,
              reward_scaling=1/10),
    pointmass=Conf(ep_len=150, num_episodes=800, eval_freq_in_episodes=40, buffer_size=100_000, lr=0.001),
    ant=Conf(ep_len=200, num_episodes=int(3e6) // 200, eval_freq_in_episodes=150,
             layer_size=512, batch_size=512, buffer_size=1_000_000)
)


class LogDeltasHistogram(BaseCallback):
    def __init__(self, env, freq_in_steps: int):
        super().__init__(verbose=0)
        self._env = env
        self._freq_in_steps = freq_in_steps

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self._freq_in_steps == 0:
            try:
                stats = self._env.envs[0].get_goal_delta_stats()
                rand_idxs = torch.randint(len(stats["deltas"]), (1000,))
                for name, val in stats.items():
                    self.logger.record(f"GoalDeltas/{name}", torch.tensor(val)[rand_idxs])
            except AttributeError:
                samples = self.model.replay_buffer.sample(1000)
                goal_fn = self._env.envs[0].achieved_goal_from_state
                deltas = goal_fn(samples.next_observations - samples.observations)
                self.logger.record("GoalDeltas/deltas", deltas[samples.dones.flatten() == 0])
        return True


class EvalCallbackSuccess(EventCallback):
    def __init__(self, eval_env, conf: Conf, log_path: str, eval_freq: int, save_fname: str, save_freq_in_episodes: int, n_eval_episodes=5):
        super().__init__(None, verbose=1)
        self._conf = conf
        self._save_fname = save_fname
        self._save_freq_in_episodes = save_freq_in_episodes
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
        do_log = self._eval_freq > 0 and self.num_timesteps % self._eval_freq == 0
        if do_log:
            self._log()

        do_save = self._eval_freq > 0 and self.num_timesteps % self._save_freq_in_episodes == 0
        if do_save:
            self.model.save(self._save_fname)
            try:
                self.model.env.envs[0].save(self._save_fname)
            except AttributeError:
                pass
            print(f"saved model in {self._save_fname}")

        return True

    def _log(self):
        rewards = []
        successes = []
        for _ in range(self._n_eval_episodes):
            results = list(eval_inflen_dict_env(model=self.model, env=self._eval_env, episode_len=self._conf.ep_len))
            rewards.append(np.mean([rew for rew, _ in results]))
            successes.append(results[-1][1]["is_success"])
        self.logger.record("DADS-MPC/is-success", np.mean(successes))
        self.logger.record("DADS-MPC/ep-mean-reward", np.mean(rewards))


def eval_cb(env, conf: Conf, save_fname: str):
    return EvalCallbackSuccess(eval_env=env, conf=conf, log_path="modelsCommandSkills",
                               eval_freq=conf.eval_freq_in_episodes*conf.ep_len, n_eval_episodes=40,
                               save_fname=save_fname, save_freq_in_episodes=100*conf.ep_len)


def get_env(name: str, drop_abs_position: bool, is_training: bool):
    dads_env_fn = envs_fns[name]
    conf: Conf = CONFS[name]

    dict_env = as_dict_env(dads_env_fn())
    dict_env = TimeLimit(dict_env, max_episode_steps=conf.ep_len)
    dict_env = TimeFeature(dict_env, is_training=is_training)
    if drop_abs_position:
        dict_env = DropGoalEnvsAbsoluteLocation(dict_env)
    return dict_env


def gamma(horizon):
    return 1 - 1/horizon


class SliderWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_dim = env.observation_space["observation"].shape[0]
        skill_dim = env.observation_space["desired_goal"].shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim + skill_dim, ))
        self._widget = create_sliders_widget(dim=skill_dim)

    def observation(self, observation):
        skill = self._widget.get_slider_values()
        return np.concatenate((observation["observation"], skill))


def main(do_render: bool, seed: int, as_gdads: bool, name: str, do_train: bool):
    drop_abs_position = True

    conf: Conf = CONFS[name]
    dict_env = get_env(name=name, drop_abs_position=drop_abs_position, is_training=True)
    if as_gdads:
        flat_env = SkillWrapper(env=dict_env)
    else:
        flat_env = flatten_env(dict_env, drop_abs_position)
    flat_env = TransformReward(flat_env, f=lambda r: r*conf.reward_scaling)
    flat_env = Monitor(flat_env)

    dict_env = get_env(name=name, drop_abs_position=drop_abs_position, is_training=False)
    if as_gdads:
        use_slider = False
        if use_slider:
            eval_env = SliderWrapper(env=dict_env)
        else:
            eval_env = GDADSEvalWrapper(dict_env, sw=BestSkillProvider(flat_env))
    else:
        eval_env = flatten_env(dict_env=dict_env, drop_abs_position=drop_abs_position)

    filename = f"modelsCommandSkills/{name}/asGDADS{as_gdads}/resamplingFalse_goalSpaceTrue-seed-{seed}"
    if os.path.exists(filename + ".zip"):
        sac = SAC.load(filename + ".zip", env=flat_env)
        print(f"loaded model {filename}")
        if as_gdads:
            flat_env.load(filename)
    else:
        sac = SAC("MlpPolicy", env=flat_env, verbose=1, learning_rate=conf.lr,
                  tensorboard_log=filename, buffer_size=conf.buffer_size, batch_size=conf.batch_size, gamma=gamma(conf.ep_len),
                  learning_starts=100*conf.ep_len, policy_kwargs=dict(log_std_init=-3, net_arch=[conf.layer_size]*2),
                  seed=seed, device="cuda", train_freq=4)
    if do_train:
        train(model=sac, conf=conf, save_fname=filename, eval_env=eval_env)
    if do_render:
        show(model=sac, env=eval_env, conf=conf)
    do_eval = not do_train and not do_render
    if do_eval:
        results = ant_grid_evaluation(model=sac, env=eval_env, episode_len=conf.ep_len)
        dump_ant_grid_evaluation(results)


def parallel_main(args):
    return main(*args)


if __name__ == '__main__':
    experiment = "ant"
    render = False
    do_train = not render
    seeds = [0]
    as_gdads = [True, False]

    args = list(product([render], seeds, as_gdads, [experiment], [do_train]))
    if len(args) == 1:
        main(*args[0])
    else:
        with Pool(processes=2) as pool:
            pool.map(parallel_main, args)