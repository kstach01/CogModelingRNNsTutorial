"""Utility functions for disentangled RNNs."""
from typing import Iterable, Callable, Any

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import warnings

from . import rnn_utils

warnings.filterwarnings("ignore")


def _get_viridis_cmap(n):
    return 


def kl_gaussian(mean: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
  r"""Calculate KL divergence between given and standard gaussian distributions.

  KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
          = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
          = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)
  Args:
    mean: mean vector of the first distribution
    var: diagonal vector of covariance matrix of the first distribution

  Returns:
    A scalar representing KL divergence of the two Gaussian distributions.
  """

  return 0.5 * jnp.sum(-jnp.log(var) - 1.0 + var + jnp.square(mean), axis=-1)


class HkDisRNN(hk.RNNCore):
  """Disentangled RNN."""

  def __init__(
      self,
      obs_size: int = 2,
      target_size: int = 1,
      latent_size: int = 10,
      update_mlp_shape: Iterable[int] = (10, 10, 10),
      choice_mlp_shape: Iterable[int] = (10, 10, 10),
      eval_mode: float = 0,
      beta_scale: int = 1,
      activation: Callable[[Any], Any] = jax.nn.relu,
  ):
    super().__init__()

    self._target_size = target_size
    self._latent_size = latent_size
    self._update_mlp_shape = update_mlp_shape
    self._choice_mlp_shape = choice_mlp_shape
    self._beta_scale = beta_scale
    self._eval_mode = eval_mode
    self._activation = activation

    # Each update MLP gets input from both the latents and the observations.
    # It has a sigma and a multiplier associated with each.
    mlp_input_size = latent_size + obs_size
    # At init the bottlenecks should all be open: sigmas small and multipliers 1
    update_mlp_sigmas_unsquashed = hk.get_parameter(
        'update_mlp_sigmas_unsquashed',
        (mlp_input_size, latent_size),
        init=hk.initializers.RandomUniform(minval=-3, maxval=-2),
    )
    # Training encourages sigmas to be ~1 or smaller. Bound them between 0 and 2
    self._update_mlp_sigmas = (
        2 * jax.nn.sigmoid(update_mlp_sigmas_unsquashed) * (1 - eval_mode)
    )
    self._update_mlp_multipliers = hk.get_parameter(
        'update_mlp_gates',
        (mlp_input_size, latent_size),
        init=hk.initializers.Constant(constant=1),
    )

    # Latents will also go through a bottleneck
    self.latent_sigmas_unsquashed = hk.get_parameter(
        'latent_sigmas_unsquashed',
        (latent_size,),
        init=hk.initializers.RandomUniform(minval=-3, maxval=-2),
    )
    self._latent_sigmas = (
        2 * jax.nn.sigmoid(self.latent_sigmas_unsquashed) * (1 - eval_mode)
    )

    # Latent initial values are also free parameters
    self._latent_inits = hk.get_parameter(
        'latent_inits',
        (latent_size,),
        init=hk.initializers.RandomUniform(minval=-0.01, maxval=0.01),
    )

  def __call__(
      self, observations: jnp.array, prev_latents: jnp.array):

    penalty = 0  # Accumulator for KL costs

    ################
    #  UPDATE MLPs #
    ################
    # Each update MLP updates one latent
    # It sees previous latents and current observation
    # It outputs a weight and an update to apply to its latent

    # update_mlp_mus_unscaled: (batch_size, obs_size + latent_size)
    update_mlp_mus_unscaled = jnp.concatenate(
        (observations, prev_latents), axis=1
    )
    # update_mlp_mus: (batch_size, obs_size + latent_size, latent_size)
    update_mlp_mus = (
        jnp.expand_dims(update_mlp_mus_unscaled, 2)
        * self._update_mlp_multipliers
    )
    # update_mlp_sigmas: (obs_size + latent_size, latent_size)
    update_mlp_sigmas = self._update_mlp_sigmas * (1 - self._eval_mode)
    # update_mlp_inputs: (batch_size, obs_size + latent_size, latent_size)
    update_mlp_inputs = update_mlp_mus + update_mlp_sigmas * jax.random.normal(
        hk.next_rng_key(), update_mlp_mus.shape
    )
    # new_latents: (batch_size, latent_size)
    new_latents = jnp.zeros(shape=(prev_latents.shape))

    # Loop over latents. Update each usings its own MLP
    for mlp_i in jnp.arange(self._latent_size):
      penalty += self._beta_scale * kl_gaussian(
          update_mlp_mus[:, :, mlp_i], update_mlp_sigmas[:, mlp_i]
      )
      update_mlp_output = hk.nets.MLP(
          self._update_mlp_shape,
          activation=self._activation,
      )(update_mlp_inputs[:, :, mlp_i])
      # update, w, new_latent: (batch_size,)
      update = hk.Linear(1,)(
          update_mlp_output
      )[:, 0]
      w = jax.nn.sigmoid(hk.Linear(1)(update_mlp_output))[:, 0]
      new_latent = w * update + (1 - w) * prev_latents[:, mlp_i]
      #new_latent = prev_latents[:, mlp_i] + update
      new_latents = new_latents.at[:, mlp_i].set(new_latent)

    #####################
    # Global Bottleneck #
    #####################
    # noised_up_latents: (batch_size, latent_size)
    noised_up_latents = new_latents + self._latent_sigmas * jax.random.normal(
        hk.next_rng_key(), new_latents.shape
    )
    penalty += kl_gaussian(new_latents, self._latent_sigmas)

    ###############
    #  CHOICE MLP #
    ###############
    # Predict targets for current time step
    # This sees previous state but does _not_ see current observation
    choice_mlp_output = hk.nets.MLP(
        self._choice_mlp_shape, activation=self._activation
    )(noised_up_latents)
    # (batch_size, target_size)
    y_hat = hk.Linear(self._target_size)(choice_mlp_output)

    # Append the penalty, so that rnn_utils can apply it as part of the loss
    penalty = jnp.expand_dims(penalty, 1)  # (batch_size, 1)
    # If we are in eval mode, there should be no penalty
    penalty = penalty * (1 - self._eval_mode)

    # output: (batch_size, target_size + 1)
    output = jnp.concatenate((y_hat, penalty), axis=1)

    return output, noised_up_latents

  def initial_state(self, batch_size):
    # (batch_size, latent_size)
    latents = jnp.ones([batch_size, self._latent_size]) * self._latent_inits
    return latents


def plot_bottlenecks(params, sort_latents=True, obs_names=None):
  """Plot the bottleneck sigmas from an hk.CompartmentalizedRNN."""
  params_disrnn = params['hk_dis_rnn']
  latent_dim = params_disrnn['latent_sigmas_unsquashed'].shape[0]
  obs_dim = params_disrnn['update_mlp_sigmas_unsquashed'].shape[0] - latent_dim

  if obs_names is None:
    if obs_dim == 2:
      obs_names = ['Choice', 'Reward']
    elif obs_dim == 5:
      obs_names = ['A', 'B', 'C', 'D', 'Reward']
    else: 
      obs_names = np.arange(1, obs_dim+1)

  latent_sigmas = 2 * jax.nn.sigmoid(
      jnp.array(params_disrnn['latent_sigmas_unsquashed'])
  )

  update_sigmas = 2 * jax.nn.sigmoid(
      np.transpose(
          params_disrnn['update_mlp_sigmas_unsquashed']
      )
  )

  if sort_latents:
    latent_sigma_order = np.argsort(
        params_disrnn['latent_sigmas_unsquashed']
    )
    latent_sigmas = latent_sigmas[latent_sigma_order]

    update_sigma_order = np.concatenate(
        (np.arange(0, obs_dim, 1), obs_dim + latent_sigma_order), axis=0
    )
    update_sigmas = update_sigmas[latent_sigma_order, :]
    update_sigmas = update_sigmas[:, update_sigma_order]

  latent_names = np.arange(1, latent_dim + 1)
  fig = plt.subplots(1, 2, figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.imshow(np.swapaxes([1 - latent_sigmas], 0, 1), cmap='Oranges')
  plt.clim(vmin=0, vmax=1)
  plt.yticks(ticks=range(latent_dim), labels=latent_names)
  plt.xticks(ticks=[])
  plt.ylabel('Latent #')
  plt.title('Latent Bottlenecks')

  plt.subplot(1, 2, 2)
  plt.imshow(1 - update_sigmas, cmap='Oranges')
  plt.clim(vmin=0, vmax=1)
  plt.colorbar()
  plt.yticks(ticks=range(latent_dim), labels=latent_names)
  xlabels = np.concatenate((np.array(obs_names), latent_names))
  plt.xticks(
      ticks=range(len(xlabels)),
      labels=xlabels,
      rotation='vertical',
  )
  plt.ylabel('Latent #')
  plt.title('Update MLP Bottlenecks')
  return fig

def plot_update_rules(params, make_network):
  """Generates visualizations of the update ruled of a disRNN.
  """

  def step(xs, state):
    core = make_network()
    output, new_state = core(jnp.expand_dims(jnp.array(xs), axis=0), state)
    return output, new_state

  _, step_hk = hk.transform(step)
  key = jax.random.PRNGKey(0)
  step_hk = jax.jit(step_hk)

  initial_state = np.array(rnn_utils.get_initial_state(make_network))
  reference_state = np.zeros(initial_state.shape)

  def plot_update_1d(params, unit_i, observations, titles):
    lim = 3
    state_bins = np.linspace(-lim, lim, 20)
    colormap = plt.cm.get_cmap('viridis', 3)
    colors = colormap.colors

    fig, ax = plt.subplots(
        1, len(observations), figsize=(len(observations) * 4, 5.5)
    )
    plt.subplot(1, len(observations), 1)
    plt.ylabel('Updated Activity')

    for observation_i in range(len(observations)):
      observation = observations[observation_i]
      plt.subplot(1, len(observations), observation_i + 1)

      plt.plot((-3, 3), (-3, 3), '--', color='grey')
      plt.plot((-3, 3), (0, 0), color='black')
      plt.plot((0, 0), (-3, 3), color='black')

      delta_states = np.zeros(shape=(len(state_bins), 1))
      for s_i in np.arange(len(state_bins)):
        state = reference_state
        state[0, unit_i] = state_bins[s_i]
        _, next_state = step_hk(
            params, key, observation, state
        )
        next_state = np.array(next_state)
        delta_states[s_i] = next_state[0, unit_i]  # - state[0, unit_i]

      plt.plot(state_bins, delta_states, color=colors[1])

      plt.title(titles[observation_i])
      plt.xlim(-lim, lim)
      plt.ylim(-lim, lim)
      plt.xlabel('Previous Activity')

      if isinstance(ax, np.ndarray):
        ax[observation_i].set_aspect('equal')
      else:
        ax.set_aspect('equal')
    return fig

  def plot_update_2d(params, unit_i, unit_input, observations, titles):
    lim = 3

    state_bins = np.linspace(-lim, lim, 20)
    colormap = plt.cm.get_cmap('viridis', len(state_bins))
    colors = colormap.colors

    fig, ax = plt.subplots(
        1, len(observations), figsize=(len(observations) * 2 + 10, 5.5)
    )
    plt.subplot(1, len(observations), 1)
    plt.ylabel('Updated Latent ' + str(unit_i + 1) + ' Activity')

    for observation_i in range(len(observations)):
      observation = observations[observation_i]
      plt.subplot(1, len(observations), observation_i + 1)

      plt.plot((-3, 3), (-3, 3), '--', color='grey')
      plt.plot((-3, 3), (0, 0), color='black')
      plt.plot((0, 0), (-3, 3), color='black')

      for si_i in np.arange(len(state_bins)):
        delta_states = np.zeros(shape=(len(state_bins), 1))
        for s_i in np.arange(len(state_bins)):
          state = reference_state
          state[0, unit_i] = state_bins[s_i]
          state[0, unit_input] = state_bins[si_i]
          _, next_state = step_hk(params, key, observation, state)
          next_state = np.array(next_state)
          delta_states[s_i] = next_state[0, unit_i]

        plt.plot(state_bins, delta_states, color=colors[si_i])

      plt.title(titles[observation_i])
      plt.xlim(-lim, lim)
      plt.ylim(-lim, lim)
      plt.xlabel('Latent ' + str(unit_i + 1) + ' Activity')

      if isinstance(ax, np.ndarray):
        ax[observation_i].set_aspect('equal')
      else:
        ax.set_aspect('equal')
    return fig

  latent_sigmas = 2*jax.nn.sigmoid(
      jnp.array(params['hk_dis_rnn']['latent_sigmas_unsquashed'])
      )
  update_sigmas = 2*jax.nn.sigmoid(
      np.transpose(
          params['hk_dis_rnn']['update_mlp_sigmas_unsquashed']
          )
      )
  latent_order = np.argsort(
      params['hk_dis_rnn']['latent_sigmas_unsquashed']
      )
  figs = []

  # Loop over latents. Plot update rules
  for latent_i in latent_order:
    # If this latent's bottleneck is open
    if latent_sigmas[latent_i] < 0.5:
      # Which of its input bottlenecks are open?
      update_mlp_inputs = np.argwhere(update_sigmas[latent_i] < 0.9)
      choice_sensitive = np.any(update_mlp_inputs == 0)
      reward_sensitive = np.any(update_mlp_inputs == 1)
      # Choose which observations to use based on input bottlenecks
      if choice_sensitive and reward_sensitive:
        observations = ([0, 0], [0, 1], [1, 0], [1, 1])
        titles = ('Left, Unrewarded',
                  'Left, Rewarded',
                  'Right, Unrewarded',
                  'Right, Rewarded')
      elif choice_sensitive:
        observations = ([0, 0], [1, 0])
        titles = ('Choose Left', 'Choose Right')
      elif reward_sensitive:
        observations = ([0, 0], [0, 1])
        titles = ('Rewarded', 'Unreward')
      else:
        observations = ([0, 0],)
        titles = ('All Trials',)
      # Choose whether to condition on other latent values
      latent_sensitive = update_mlp_inputs[update_mlp_inputs > 1] - 2
      # Doesn't count if it depends on itself (this'll be shown no matter what)
      latent_sensitive = np.delete(
          latent_sensitive, latent_sensitive == latent_i
      )
      if not latent_sensitive.size:  # Depends on no other latents
        fig = plot_update_1d(params, latent_i, observations, titles)
      else:  # It depends on latents other than itself.
        fig = plot_update_2d(
            params,
            latent_i,
            latent_sensitive[np.argmax(latent_sensitive)],
            observations,
            titles,
        )
      if len(latent_sensitive) > 1:
        print(
            'WARNING: This update rule depends on more than one '
            + 'other latent. Plotting just one of them'
        )

      figs.append(fig)

  return figs
