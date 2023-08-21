from typing import Iterable, Callable, Any

import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp

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
        init=hk.initializers.RandomUniform(minval=-1, maxval=1),
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
