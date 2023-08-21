from __future__ import print_function

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import chex
import matplotlib.pyplot as plt

class DatasetRNN():
  """Holds a dataset for training an RNN, consisting of inputs and targets.

     Both inputs and targets are stored as [timestep, episode, feature]
     Serves them up in batches
  """

  def __init__(self,
               xs: np.ndarray,
               ys: np.ndarray,
               batch_size: Optional[int] = None):
    """Do error checking and bin up the dataset into batches.

    Args:
      xs: Values to become inputs to the network.
        Should have dimensionality [timestep, episode, feature]
      ys: Values to become output targets for the RNN.
        Should have dimensionality [timestep, episode, feature]
      batch_size: The size of the batch (number of episodes) to serve up each
        time next() is called. If not specified, all episodes in the dataset 
        will be served

    """

    if batch_size is None:
      batch_size = xs.shape[1]

    # Error checking
    # Do xs and ys have the same number of timesteps?
    if xs.shape[0] != ys.shape[0]:
      msg = ('number of timesteps in xs {} must be equal to number of timesteps'
             ' in ys {}.')
      raise ValueError(msg.format(xs.shape[0], ys.shape[0]))

    # Do xs and ys have the same number of episodes?
    if xs.shape[1] != ys.shape[1]:
      msg = ('number of timesteps in xs {} must be equal to number of timesteps'
             ' in ys {}.')
      raise ValueError(msg.format(xs.shape[0], ys.shape[0]))

    # Is the number of episodes divisible by the batch size?
    if xs.shape[1] % batch_size != 0:
      msg = 'dataset size {} must be divisible by batch_size {}.'
      raise ValueError(msg.format(xs.shape[1], batch_size))

    # Property setting
    self._xs = xs
    self._ys = ys
    self._batch_size = batch_size
    self._dataset_size = self._xs.shape[1]
    self._idx = 0
    self.n_batches = self._dataset_size // self._batch_size

  def __iter__(self):
    return self

  def __next__(self):
    """Return a batch of data, including both xs and ys."""

    # Define the chunk we want: from idx to idx + batch_size
    start = self._idx
    end = start + self._batch_size
    # Check that we're not trying to overshoot the size of the dataset
    assert end <= self._dataset_size

    # Update the index for next time
    if end == self._dataset_size:
      self._idx = 0
    else:
      self._idx = end

    # Get the chunks of data
    x, y = self._xs[:, start:end], self._ys[:, start:end]

    return x, y


def nan_in_dict(d):
  """Check a nested dict (e.g. hk.params) for nans."""
  if not isinstance(d, dict):
    return np.any(np.isnan(d))
  else:
    return any(nan_in_dict(v) for v in d.values()) 
    

def train_model(
    model_fun: Callable[[], hk.RNNCore],
    dataset: DatasetRNN,
    optimizer: optax.GradientTransformation = optax.adam(1e-3),
    random_key: Optional[chex.PRNGKey] = None,
    opt_state: Optional[optax.OptState] = None,
    params: Optional[hk.Params] = None,
    n_steps: int = 1000,
    penalty_scale=0,
    loss: str = 'categorical',
    do_plot: bool = True,
) -> Tuple[hk.Params, optax.OptState, Dict[str, np.ndarray]]:
  """Trains a model for a fixed number of steps.

  Args:
    model_fun: A function that, when called, returns a Haiku RNN object
    training_dataset: A DatasetRNN, containing the data you wish to train on
    opt: The optimizer you'd like to use to train the network
    random_key: A jax random key, to be used in initializing the network
    opt_state: An optimzier state suitable for opt
      If not specified, will initialize a new optimizer from scratch
    params:  A set of parameters suitable for the network given by make_network
      If not specified, will begin training a network from scratch
    n_steps: An integer giving the number of steps you'd like to train for
    penalty_scale:
    loss:
    do_plot: Boolean that controls whether a learning curve is plotted

  Returns:
    params: Trained parameters
    opt_state: Optimizer state at the end of training
    losses: Losses on both datasets
  """

  sample_xs, _ = next(dataset)  # Get a sample input, for shape

  # Haiku, step one: Define the batched network
  def unroll_network(xs):
    core = model_fun()
    batch_size = jnp.shape(xs)[1]
    state = core.initial_state(batch_size)
    ys, _ = hk.dynamic_unroll(core, xs, state)
    return ys

  # Haiku, step two: Transform the network into a pair of functions
  # (model.init and model.apply)
  model = hk.transform(unroll_network)

  # PARSE INPUTS
  if random_key is None:
    random_key = jax.random.PRNGKey(0)
  # If params have not been supplied, start training from scratch
  if params is None:
    random_key, key1 = jax.random.split(random_key)
    params = model.init(key1, sample_xs)
  # It an optimizer state has not been supplied, start optimizer from scratch
  if opt_state is None:
    opt_state = optimizer.init(params)

  def categorical_log_likelihood(
      labels: np.ndarray, output_logits: np.ndarray
  ) -> float:
    # Mask any errors for which label is negative
    mask = jnp.logical_not(labels < 0)
    log_probs = jax.nn.log_softmax(output_logits)
    if labels.shape[2] != 1:
      raise ValueError(
          'Categorical loss function requires targets to be of dimensionality'
          ' (n_timesteps, n_episodes, 1)'
      )
    one_hot_labels = jax.nn.one_hot(
        labels[:, :, 0], num_classes=output_logits.shape[-1]
    )
    log_liks = one_hot_labels * log_probs
    masked_log_liks = jnp.multiply(log_liks, mask)
    loss = -jnp.nansum(masked_log_liks)
    return loss

  def categorical_loss(
      params, xs: np.ndarray, labels: np.ndarray, random_key
  ) -> float:
    output_logits = model.apply(params, random_key, xs)
    loss = categorical_log_likelihood(labels, output_logits)
    return loss

  def penalized_categorical_loss(
      params, xs, targets, random_key, penalty_scale=penalty_scale
  ) -> float:
    """Treats the last element of the model outputs as a penalty."""
    # (n_steps, n_episodes, n_targets)
    model_output = model.apply(params, random_key, xs)
    output_logits = model_output[:, :, :-1]
    penalty = jnp.sum(model_output[:, :, -1])  # ()
    loss = (
        categorical_log_likelihood(targets, output_logits)
        + penalty_scale * penalty
    )
    return loss

  losses = {
      'categorical': categorical_loss,
      'penalized_categorical': penalized_categorical_loss,
  }
  compute_loss = jax.jit(losses[loss])

  # Define what it means to train a single step
  @jax.jit
  def train_step(
      params, opt_state, xs, ys, random_key
  ) -> Tuple[float, Any, Any]:
    loss, grads = jax.value_and_grad(compute_loss, argnums=0)(
        params, xs, ys, random_key
    )
    grads, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, grads)
    return loss, params, opt_state

  # Train the network!
  training_loss = []
  for step in jnp.arange(n_steps):
    random_key, key1, key2 = jax.random.split(random_key, 3)
    # Train on training data
    xs, ys = next(dataset)
    loss, params, opt_state = train_step(params, opt_state, xs, ys, key2)
    # Log every 10th step
    if step % 10 == 9:
      training_loss.append(float(loss))
      print((f'\rStep {step + 1} of {n_steps}; '
             f'Loss: {loss:.4e}. '), end='')

  # If we actually did any training, print final loss and make a nice plot
  if n_steps > 1 and do_plot:

    plt.figure()
    plt.semilogy(training_loss, color='black')
    plt.xlabel('Training Step')
    plt.ylabel('Mean Loss')
    plt.title('Loss over Training')

  losses = {
      'training_loss': np.array(training_loss),
  }

  # Check if anything has become NaN that should not be NaN
  if nan_in_dict(params):
    print(params)
    raise ValueError('NaN in params')
  if len(training_loss) > 0 and np.isnan(training_loss[-1]):
    raise ValueError('NaN in loss')

  return params, opt_state, losses


def fit_model(
    model_fun,
    dataset,
    optimizer=None,
    loss:str='categorical',
    convergence_thresh=1e-5,
    random_key=None,
):
  """Fits a model to convergence, by repeatedly calling train_model.
  """

  if random_key is None:
    random_key = jax.random.PRNGKey(0)

  # Initialize the model
  params, opt_state, losses = train_model(
        model_fun,
        dataset,
        optimizer=optimizer,
        n_steps=0,
    )
  
  # Train until the loss stops going down
  converged = False
  loss = np.inf
  while not converged:
    params, opt_state, losses = train_model(
        model_fun,
        dataset,
        params=params,
        opt_state=opt_state,
        optimizer=optimizer,
        loss = loss,
        do_plot = False,
        n_steps=1000,
    )

    loss_new = losses['training_loss'][-1]
    # Declare "converged" if loss has not improved very much (but has improved)
    converged = loss_new < loss * (1+convergence_thresh) and loss_new > loss * (1 - convergence_thresh)
    if converged:
      print('Model Converged!')
    else:
      print('Model not yet converged - Running more steps of gradient descent.')
    loss = loss_new
      
  return params, loss

def eval_model(
    model_fun: Callable[[], hk.RNNCore],
    params: hk.Params,
    xs: np.ndarray,
) ->  Tuple[np.ndarray, Any]:
  """Run an RNN with specified params and inputs. Track internal state.

  Args:
    make_network: A Haiku function that defines a network architecture
    params: A set of params suitable for that network
    xs: A batch of inputs [timesteps, episodes, features] suitable for the model

  Returns:
    y_hats: Network outputs at each timestep
    states: Network states at each timestep
  """

  n_steps = jnp.shape(xs)[0]

  def unroll_network(xs):
    core = model_fun()
    batch_size = jnp.shape(xs)[1]
    state = core.initial_state(batch_size)

    y_hats = []
    states = []

    for t in range(n_steps):
      states.append(state)
      y_hat, new_state = core(xs[t, :], state)
      state = new_state

      y_hats.append(y_hat)

    return np.asarray(y_hats), np.asarray(states)

  model = hk.transform(unroll_network)
  key = jax.random.PRNGKey(np.random.randint(2**32))
  y_hats, states = model.apply(params, key, xs)

  return np.asarray(y_hats), states
