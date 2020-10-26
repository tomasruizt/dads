import os
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


class DensityEstimator:
    def __init__(self, input_dim: int, vae_training_batch_size: int,
                 samples_generator: Callable[[int], np.ndarray]):
        latent_space_dim = 6
        layer_size = 256
        activation = "tanh"

        prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(latent_space_dim))
        self._encoder = encoder = tfk.Sequential([
            tfk.layers.InputLayer(input_shape=[input_dim]),
            tfk.layers.Dense(units=layer_size, activation=activation),
            tfk.layers.Dense(units=layer_size, activation=activation),
            tfk.layers.Dense(units=layer_size, activation=activation),
            tfk.layers.Dense(units=tfp.layers.MultivariateNormalTriL.params_size(latent_space_dim)),
            tfp.layers.MultivariateNormalTriL(
                event_size=latent_space_dim,
                activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.01, use_exact_kl=True),
            )
        ])

        self._decoder = decoder = tfk.Sequential([
            tfk.layers.InputLayer(input_shape=[latent_space_dim]),
            tfk.layers.Dense(units=layer_size, activation=activation),
            tfk.layers.Dense(units=layer_size, activation=activation),
            tfk.layers.Dense(units=layer_size, activation=activation),
            tfk.layers.Dense(units=input_dim),
        ])

        self.VAE = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))
        self.VAE.compile(loss="mse")

        self._samples_generator = samples_generator

    @tf.function(experimental_relax_shapes=True)
    def get_input_density(self, x: np.ndarray):
        zs = self._encoder(x).mean()
        approx_latent_distr = self._encoder(self._samples_generator(500))
        return tf.reduce_mean(approx_latent_distr.prob(zs[..., None, :]), axis=1)


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


def viz_grid_probabilites(prob_fn: Callable[[np.ndarray], np.ndarray]):
    plt.ion()
    grid_granularity = 100
    X, Y = np.mgrid[-2:2:complex(0, grid_granularity), -1:1:complex(0, grid_granularity)]
    pts = np.asarray([X.ravel(), Y.ravel()]).T

    pts_probs = prob_fn(pts)

    Z = pts_probs.reshape(grid_granularity, grid_granularity)
    fig, ax = plt.subplots()
    ax.contourf(X, Y, Z, cmap='Blues', zorder=-1)
    ax.contour(X, Y, Z, zorder=-1)
    fig.canvas.draw()
    fig.canvas.flush_events()


if __name__ == '__main__':
    env = LongishEnv()
    state_space_dim = env.observation_space.shape[0]
    batch_size = 256

    buffer = collect_samples(env=env, num_episodes=1000)

    model = DensityEstimator(
        input_dim=state_space_dim,
        vae_training_batch_size=batch_size,
        samples_generator=lambda n: sample(buffer, n)
    )

    def viz_callback(epoch, *args):
        if epoch % 5 == 0:
            viz_grid_probabilites(prob_fn=lambda pts: model.get_input_density(pts).numpy())

    model_filename = "density-model-ckpt/"
    if os.path.exists(model_filename):
        model.VAE.load_weights(filepath=model_filename)

    train = True
    if train:
        cb = tf.keras.callbacks.ModelCheckpoint(filepath=model_filename, save_weights_only=True)
        cb2 = tf.keras.callbacks.LambdaCallback(on_epoch_end=viz_callback)
        model.VAE.fit(x=buffer, y=buffer, batch_size=batch_size, epochs=100, validation_split=0.2, callbacks=[cb, cb2])

    buffer_samples = buffer[np.random.choice(len(buffer), 2000)]
    state_pred = model.VAE(buffer_samples)
    env.render("human", more_pts={"blue": buffer_samples, "orange": state_pred.numpy()})

    input("Exit")
