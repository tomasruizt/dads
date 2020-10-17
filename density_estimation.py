import os
import time

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import Env
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as tfk
from tensorflow_probability.python.distributions import Normal, MultivariateNormalTriL


class LongishEnv(Env):
    _corners = np.asarray([(-2, -1), (2, -1), (2, 1), (-2, 1), (-2, -1)])
    _initial_positions = np.asarray([(-1.8, 0), (1.8, 0)])
    action_space = gym.spaces.Box(low=-0.2, high=0.2, shape=(2,))

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


def new_VAE(state_space_dim: int, latent_space_dim: int, batch_size: int):
    layer_size = 64
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

    return VAE, encoder, decoder


def collect_samples(num_episodes: int):
    buffer = []
    for _ in range(num_episodes):
        env.reset()
        for _ in range(50):
            obs, *_ = env.step(env.action_space.sample())
            buffer.append(obs)
    return np.asarray(buffer)


if __name__ == '__main__':
    state_space_dim = 2
    latent_space_dim = 6
    batch_size = 32

    VAE, encoder, decoder = new_VAE(state_space_dim=state_space_dim,
                                    latent_space_dim=latent_space_dim,
                                    batch_size=batch_size)
    model_filename = "density-model-ckpt/"
    if os.path.exists(model_filename):
        VAE.load_weights(filepath=model_filename)

    env = LongishEnv()
    train = True
    buffer = collect_samples(num_episodes=1000)

    if train:
        ds_size = len(buffer)
        cb = tf.keras.callbacks.ModelCheckpoint(filepath=model_filename, save_weights_only=True)
        VAE.fit(x=buffer, y=buffer, batch_size=batch_size, epochs=3, validation_split=0.2, callbacks=cb)

    buffer_samples = buffer[np.random.choice(len(buffer), 500)]

    def sample():
        return buffer[np.random.choice(len(buffer), 10)]

    pred = VAE(buffer_samples)
    env.render("human", more_pts={"blue": buffer_samples, "orange": pred.numpy()})

    grid_granularity = 40
    X, Y = np.mgrid[-2:2:complex(0, grid_granularity), -1:1:complex(0, grid_granularity)]
    pts = np.asarray([X.ravel(), Y.ravel()]).T

    support = encoder(buffer[np.random.choice(len(buffer), 100)])
    probs = np.asarray([tf.reduce_mean(support.prob(z)) for z in encoder(pts).mean()])

    Z = probs.reshape(grid_granularity, grid_granularity)
    plt.contourf(X, Y, Z, cmap='Blues', zorder=-1)
    plt.contour(X, Y, Z, zorder=-1)
    plt.show()

    input("Exit")