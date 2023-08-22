"""Define bi-RNNs.

https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks
"""
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp

RNNState = jnp.array


class BiRNN(hk.RNNCore):
  """A bifurcating RNN: "habit" processes action sequences; "value" rewards."""

  def __init__(self, rl_params, network_params, init_value=0.5):

    super().__init__()

    # self._hs = rl_params.hs
    # self._ho = rl_params.ho
    # self._vs = rl_params.vs
    # self._vo = rl_params.vo

    self._zero_values = rl_params.zero_values
    self._zero_v_states = rl_params.zero_v_states

    self._hs = rl_params.s
    self._vs = rl_params.s
    self._ho = rl_params.o
    self._vo = rl_params.o

    self._n_actions = network_params.n_actions
    self._hidden_size = network_params.hidden_size
    self._final_activation_fn = network_params.final_activation_fn
    self.beta = rl_params.beta

    if rl_params['fit_init_v']:
      init = hk.initializers.RandomNormal(stddev=1, mean=1)
      self._init_value_v = hk.get_parameter('init_value_v', (1,), init=init)
    else:
      self._init_value_v = init_value

    if rl_params['fit_init_h']:
      init = hk.initializers.RandomNormal(stddev=1, mean=1)
      self._init_value_h = hk.get_parameter('init_value_h', (1,), init=init)
    else:
      self._init_value_h = init_value

    if rl_params['fit_forget']:
      init = hk.initializers.RandomNormal(stddev=1, mean=0)
      self.forget = jax.nn.sigmoid(  # 0 < forget < 1
          hk.get_parameter('unsigmoid_forget', (1,), init=init)
      )
    else:
      self.forget = rl_params['forget']

    self._w_v = rl_params.w_v
    self._w_h = rl_params.w_h

  def _value_rnn(self, state, value, action, reward):

    pre_act_val = jnp.sum(value * action, axis=1)  # (batch_s, 1)
    if self._zero_values:
      pre_act_val = jnp.zeros_like(pre_act_val)  # zero input value
    if self._zero_v_states:
      state = jnp.zeros_like(state)  # zero input state

    inputs = jnp.concatenate(
        [pre_act_val[:, jnp.newaxis], reward[:, jnp.newaxis]], axis=-1)
    if self._vo:  # "o" = output -> feed previous output back in
      inputs = jnp.concatenate([inputs, value], axis=-1)
    if self._vs:  # "s" = state -> feed previous hidden state back in
      inputs = jnp.concatenate([inputs, state], axis=-1)

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))

    update = hk.Linear(1)(next_state)
    # Forget non-chosen values / preferences: decay tow. curr. mean value
    # value = (1 - self.forget) * value + self.forget * jnp.mean(
        # value, axis=1, keepdims=True)
    # Forget non-chosen values / preferences: decay tow. init value
    value = (1 - self.forget) * value + self.forget * self._init_value_v
    if self._zero_values:
      next_value = (1 - action) * value + action * update  # zero input value
    else:
      next_value = value + action * update

    return next_value, next_state

  def _habit_rnn(self, state, habit, action):

    inputs = action
    if self._ho:  # "o" = output -> feed previous output back in
      inputs = jnp.concatenate([inputs, habit], axis=-1)
    if self._hs:  # "s" = state -> feed previous hidden state back in
      inputs = jnp.concatenate([inputs, state], axis=-1)

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))
    next_habit = hk.Linear(self._n_actions)(next_state)

    return next_habit, next_state

  def __call__(self, inputs: jnp.ndarray, prev_state: jnp.ndarray):
    h_state, v_state, habit, value = prev_state
    action = inputs[:, :self._n_actions]  # shape: (batch_size, n_actions)
    reward = inputs[:, -1]  # shape: (batch_size,)

    # Value module: update/create new values
    next_value, next_v_state = self._value_rnn(v_state, value, action, reward)

    # Habit module: update/create new habit
    next_habit, next_h_state = self._habit_rnn(h_state, habit, action)

    # Combine value and habit
    hv_combo = self._w_v * next_value + self._w_h * next_habit  # (bs, n_a)
    action_probs = self._final_activation_fn(self.beta * hv_combo)  # (bs, n_a)

    return action_probs, (next_h_state, next_v_state, next_habit, next_value)

  def initial_state(self, batch_size: Optional[int]):

    return (
        0 * jnp.ones([batch_size, self._hidden_size]),  # h_state
        0 * jnp.ones([batch_size, self._hidden_size]),  # v_state
        self._init_value_h * jnp.ones([batch_size, self._n_actions]),  # habit
        self._init_value_v * jnp.ones([batch_size, self._n_actions]),  # value
        )