import os
import time
from functools import partial
from typing import Callable

import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from gym import Env
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as tfk

from envs.fetch import make_fetch_pick_and_place_env


class LongishEnv(Env):
    _corners = np.asarray([(-2, -1), (2, -1), (2, 1), (-2, 1), (-2, -1)])
    _initial_positions = np.asarray([(-1.8, 0), (1.8, 0)])
    action_space = gym.spaces.Box(low=-0.2, high=0.2, shape=(2,))
    observation_space = gym.spaces.Box(low=np.asarray([-2, -1]), high=np.asarray([2, 1]))

    def __init__(self):
        self._plot = None
        self._agent_pos = self._initial_positions[0].copy()

    def step(self, action):
        self._agent_pos += action
        self._agent_pos = np.clip(self._agent_pos, a_min=[-2, -1], a_max=[2, 1])
        reward, done, info = 0, False, None
        return self._agent_pos.copy(), reward, done, info

    def reset(self):
        self._agent_pos = self._initial_positions[np.random.randint(2)].copy()
        return self._agent_pos.copy()

    def render(self, mode='human', more_pts=None):
        if self._plot is None:
            plt.ion() if mode == "human" else plt.ioff()
            fig, ax = plt.subplots()
            ax.plot(*self._corners.T)
            path_collection = ax.scatter(*self._agent_pos, c="red")
            self._plot = fig, ax, path_collection
        fig, ax, path_collection = self._plot
        path_collection.set_offsets(self._agent_pos)
        if more_pts:
            for color, pts in more_pts.items():
                ax.scatter(*pts.T, c=color)
        fig.canvas.draw()
        fig.canvas.flush_events()


def new_VAE(state_space_dim: int, batch_size: int,
            samples_generator: Callable[[int], np.ndarray]):
    latent_space_dim = 6
    layer_size = 256
    activation = "tanh"

    prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(latent_space_dim))
    encoder = tfk.Sequential([
        tfk.layers.InputLayer(input_shape=[state_space_dim]),
        tfk.layers.Dense(units=layer_size, activation=activation),
        tfk.layers.Dense(units=layer_size, activation=activation),
        tfk.layers.Dense(units=layer_size, activation=activation),
        tfk.layers.Dense(units=tfp.layers.MultivariateNormalTriL.params_size(latent_space_dim)),
        tfp.layers.MultivariateNormalTriL(
            event_size=latent_space_dim,
            activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1/batch_size, use_exact_kl=True),
        )
    ])

    decoder = tfk.Sequential([
        tfk.layers.InputLayer(input_shape=[latent_space_dim]),
        tfk.layers.Dense(units=layer_size, activation=activation),
        tfk.layers.Dense(units=layer_size, activation=activation),
        tfk.layers.Dense(units=layer_size, activation=activation),
        tfk.layers.Dense(units=state_space_dim),
    ])

    VAE = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))
    VAE.compile(loss="mse")

    mean = tf.reduce_mean

    @tf.function
    def get_state_density(s: np.ndarray):
        support = encoder(samples_generator(500))

        @tf.function
        def prob(z):
            return mean(support.prob(z))

        return tf.map_fn(prob, encoder(s).mean())

    return VAE, get_state_density


def collect_samples(env, num_episodes: int):
    samples = []
    for _ in range(num_episodes):
        env.reset()
        for _ in range(50):
            obs, *_ = env.step(env.action_space.sample())
            samples.append(obs)
    return np.asarray(samples)


def sample(array: np.ndarray, n: int) -> np.ndarray:
    indices = np.random.choice(len(array), n)
    return array[indices]


def viz_probabilites(states: np.ndarray, actual_probs: np.ndarray, pred_probs: np.ndarray):
    fig, axs = plt.subplots(2, 1)
    axs[0].tricontourf(*states.T, actual_probs)
    axs[1].tricontourf(*states.T, pred_probs)
    fig.canvas.draw()
    fig.canvas.flush_events()


if __name__ == '__main__':
    env = LongishEnv()
    state_space_dim = env.observation_space.shape[0]
    batch_size = 32

    buffer = collect_samples(env=env, num_episodes=1000)

    VAE, get_state_density = new_VAE(
        state_space_dim=state_space_dim,
        batch_size=batch_size,
        samples_generator=lambda n: sample(buffer, n)
    )
    model_filename = "density-model-ckpt/"
    if os.path.exists(model_filename):
        VAE.load_weights(filepath=model_filename)

    train = True

    if train:
        ds_size = len(buffer)
        cb = tf.keras.callbacks.ModelCheckpoint(filepath=model_filename, save_weights_only=True)
        VAE.fit(x=buffer, y=[buffer, buffer], batch_size=batch_size, epochs=10, validation_split=0.2, callbacks=cb)

    buffer_samples = buffer[np.random.choice(len(buffer), 2000)]
    state_pred = VAE(buffer_samples)
    env.render("human", more_pts={"blue": buffer_samples, "orange": state_pred.numpy()})

    grid_granularity = 100
    X, Y = np.mgrid[-2:2:complex(0, grid_granularity), -1:1:complex(0, grid_granularity)]
    pts = np.asarray([X.ravel(), Y.ravel()]).T

    pts_probs = get_state_density(pts).numpy()

    Z = pts_probs.reshape(grid_granularity, grid_granularity)
    plt.contourf(X, Y, Z, cmap='Blues', zorder=-1)
    plt.contour(X, Y, Z, zorder=-1)
    plt.show()

    input("Exit")
    exit()
