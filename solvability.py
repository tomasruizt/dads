import os
from collections import OrderedDict

import gym
from gym import ObservationWrapper
from gym.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.her.her import HER
from stable_baselines3.sac.sac import SAC
import numpy as np

from envs.custom_envs import make_fetch_reach_env, make_fetch_pick_and_place_env


class TimeFeature(ObservationWrapper):
    def __init__(self, env: TimeLimit):
        super().__init__(env)
        self._time_feature_range = (-1, 1)
        obs_len = self.env.observation_space["observation"].shape[0]
        self.observation_space = gym.spaces.Dict(spaces=dict(
            achieved_goal=env.observation_space["achieved_goal"],
            desired_goal=env.observation_space["desired_goal"],
            observation=gym.spaces.Box(low=np.asarray([*[-np.inf]*obs_len, min(self._time_feature_range)]),
                                       high=np.asarray([*[np.inf]*obs_len, max(self._time_feature_range)]))
        ))

    def observation(self, observation: dict):
        obs = OrderedDict(**observation)
        obs["observation"] = np.hstack((obs["observation"], self._get_time_feature()))
        return obs

    def _get_time_feature(self):
        self.env: TimeLimit
        fmax, fmin = max(self._time_feature_range), min(self._time_feature_range)
        tmax, tmin, t = self.env._max_episode_steps, 0, self.env._elapsed_steps
        f = ((t - tmin) / (tmax - tmin)) * (fmin - fmax) + fmax
        assert fmin <= f <= fmax, (f, self.env._elapsed_steps)
        return f


class ForHER(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        goal_space = gym.spaces.Box(-np.inf, np.inf, shape=(len(self.env.get_goal()),))
        self.observation_space = gym.spaces.Dict(spaces=dict(
            achieved_goal=goal_space, desired_goal=goal_space, observation=self.env.observation_space
        ))

    def observation(self, observation):
        return OrderedDict(achieved_goal=self.env.achieved_goal_from_state(observation),
                           desired_goal=self.env.get_goal(),
                           observation=observation)


if __name__ == '__main__':
    env = make_fetch_pick_and_place_env()
    episode_len = 50
    env = ForHER(env=env)
    env = TimeLimit(env=env, max_episode_steps=episode_len)
    env = TimeFeature(env=env)
    env = Monitor(env=env)

    model_file = "./her-results/first-run/model"
    if os.path.exists(model_file + ".zip"):
        model = HER.load(model_file, env=env)
        print(f"Loaded from {model_file}")
    else:
        model = HER(policy="MlpPolicy", env=env, model_class=SAC, max_episode_length=episode_len,
                    online_sampling=True, tensorboard_log="./her-results/")
    model.learn(int(1e6), tb_log_name="./snd-run")
    model.save(model_file)

    while True:
        obs = env.reset()
        for _ in range(episode_len):
            action, _ = model.predict(obs, deterministic=True)
            obs, *_ = env.step(action)
            env.render("human")
