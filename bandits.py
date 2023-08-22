"""Environments + agents for 2-armed bandit task."""
# pylint: disable=line-too-long
from typing import Callable, NamedTuple, Tuple, Union, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from CogModelingRNNsTutorial.rnn_utils import DatasetRNN

###################################
# CONVENIENCE FUNCTIONS.          #
###################################


def _check_in_0_1_range(x, name):
  if not (0 <= x <= 1):
    raise ValueError(
        f'Value of {name} must be in [0, 1] range. Found value of {x}.')


###################################
# GENERATIVE FUNCTIONS FOR AGENTS #
###################################


class AgentQ:
  """An agent that runs simple Q-learning for the y-maze tasks.

  Attributes:
    alpha: The agent's learning rate
    beta: The agent's softmax temperature
    q: The agent's current estimate of the reward probability on each arm
  """

  def __init__(
      self,
      alpha: float = 0.2,
      beta: float = 3.,
      n_actions: int = 2,
      forgetting_rate: float = 0.):
    """Update the agent after one step of the task.

    Args:
      alpha: scalar learning rate
      beta: scalar softmax inverse temperature parameter.
      n_actions: number of actions (default=2)
      forgetting_rate: rate at which q values decay toward the initial values
    """
    self._alpha = alpha
    self._beta = beta
    self._n_actions = n_actions
    self._forgetting_rate = forgetting_rate
    self._q_init = 0.5
    self.new_sess()

    _check_in_0_1_range(alpha, 'alpha')
    _check_in_0_1_range(alpha, 'alpha')

  def new_sess(self):
    """Reset the agent for the beginning of a new session."""
    self._q = self._q_init * np.ones(self._n_actions)

  def get_choice_probs(self) -> np.ndarray:
    """Compute the choice probabilities as softmax over q."""
    choice_probs = np.exp(self._beta * self._q) / np.sum(
        np.exp(self._beta * self._q))
    return choice_probs

  def get_choice(self) -> int:
    """Sample a choice, given the agent's current internal state."""
    choice_probs = self.get_choice_probs()
    choice = np.random.choice(self._n_actions, p=choice_probs)
    return choice

  def update(self,
             choice: int,
             reward: int):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent. 0 or 1
    """
    # Decay q-values toward the initial value.
    self._q = ((1-self._forgetting_rate) * self._q +
               self._forgetting_rate * self._q_init)

    # Update chosen q for chosen action with observed reward.
    self._q[choice] = (1 - self._alpha) * self._q[choice] + self._alpha * reward

  @property
  def q(self):
    # This establishes q as an externally visible attribute of the agent.
    # For agent = AgentQ(...), you can view the q values with agent.q; however,
    # you will not be able to modify them directly because you will be viewing
    # a copy.
    return self._q.copy()


class AgentNetwork:
  """A class that allows running a pretrained RNN as an agent.

  Attributes:
    make_network: A Haiku function that returns an RNN architecture
    params: A set of Haiku parameters suitable for that architecture
  """

  def __init__(self,
               make_network: Callable[[], hk.RNNCore],
               params: hk.Params,
               n_actions: int = 2):
    """Initialize the agent network.
    
    Args: 
      make_network: function that instantiates a callable haiku network object
      params: parameters for the network
      n_actions: number of permitted actions (default = 2)
    """

    def _step_network(xs: np.ndarray,
                      state: hk.State) -> Tuple[np.ndarray, hk.State]:
      """Apply one step of the network.
      
      Args:
        xs: array containing network inputs
        state: previous state of the hidden units of the RNN model.
        
      Returns:
        y_hat: output of RNN
        new_state: state of the hidden units of the RNN model
      """
      core = make_network()
      y_hat, new_state = core(xs, state)
      return y_hat, new_state

    def _get_initial_state() -> hk.State:
      """Get the initial state of the hidden units of RNN model."""
      core = make_network()
      state = core.initial_state(1)
      return state

    key = jax.random.PRNGKey(0)
    model = hk.transform(_step_network)
    state = hk.transform(_get_initial_state)

    self._initial_state = state.apply(params, key)
    self._model_fun = jax.jit(
        lambda xs, state: model.apply(params, key, xs, state))
    self._xs = np.zeros((1, 2))
    self._n_actions = n_actions
    self.new_sess()

  def new_sess(self):
    """Reset the network for the beginning of a new session."""
    self._state = self._initial_state

  def get_choice_probs(self) -> np.ndarray:
    """Predict the choice probabilities as a softmax over output logits."""
    output_logits, _ = self._model_fun(self._xs, self._state)
    output_logits = np.array(output_logits)
    output_logits = output_logits[0][:self._n_actions]
    choice_probs = np.exp(output_logits) / np.sum(
        np.exp(output_logits))
    return choice_probs

  def get_choice(self) -> Tuple[int, np.ndarray]:
    """Sample choice."""
    choice_probs = self.get_choice_probs()
    choice = np.random.choice(self._n_actions, p=choice_probs)
    return choice

  def update(self, choice: int, reward: int):
    self._xs = np.array([[choice, reward]])
    _, new_state = self._model_fun(self._xs, self._state)
    self._state = np.array(new_state)


################
# ENVIRONMENTS #
################


class EnvironmentBanditsFlips:
  """Env for 2-armed bandit task with with reward probs that flip in blocks."""

  def __init__(
      self,
      block_flip_prob: float = 0.02,
      reward_prob_high: float = 0.8,
      reward_prob_low: float = 0.2,
  ):
    # Assign the input parameters as properties
    self._block_flip_prob = block_flip_prob
    self._reward_prob_high = reward_prob_high
    self._reward_prob_low = reward_prob_low
    # Choose a random block to start in
    self._block = np.random.binomial(1, 0.5)
    # Set up the new block
    self.new_block()

  def new_block(self):
    """Flip the reward probabilities for a new block."""
    # Flip the block
    self._block = 1 - self._block
    # Set the reward probabilites
    if self._block == 1:
      self.reward_probs = [self._reward_prob_high, self._reward_prob_low]
    else:
      self.reward_probs = [self._reward_prob_low, self._reward_prob_high]

  def step(self, choice):
    """Step the model forward given chosen action."""
    # Choose the reward probability associated with agent's choice
    reward_prob_trial = self.reward_probs[choice]

    # Sample a reward with this probability
    reward = float(np.random.binomial(1, reward_prob_trial))

    # Check whether to flip the block
    if np.random.binomial(1, self._block_flip_prob):
      self.new_block()

    # Return the reward
    return float(reward)

  @property
  def n_actions(self) -> int:
    return 2


class VanillaAgentQ(AgentQ):
  """This agent is a wrapper of AgentQ with only alpha and beta parameters."""

  def __init__(self, alpha: float, beta: float, n_actions: int = 2):
    super(VanillaAgentQ, self).__init__(
        alpha, beta, n_actions=n_actions, forgetting_rate=0.)


class MysteryAgentQ(AgentQ):
  """Don't look at this agent if you want to do the exercises!"""

  def __init__(self, alpha: float, beta: float, n_actions: int = 2):
    super(MysteryAgentQ, self).__init__(
        alpha, beta, n_actions=n_actions, forgetting_rate=0.1)


class EnvironmentBanditsDrift:
  """Environment for a drifting two-armed bandit task.

  Reward probabilities on each arm are sampled randomly between 0 and 1. On each
  trial, gaussian random noise is added to each.

  Attributes:
    sigma: A float, between 0 and 1, giving the magnitude of the drift
    reward_probs: Probability of reward associated with each action
    n_actions: number of actions available
  """

  def __init__(
      self,
      sigma: float,
      n_actions: int = 2,
      ):
    """Initialize the environment."""
    # Check inputs
    if sigma < 0:
      msg = f'Argument sigma but must be greater than 0. Found: {sigma}.'
      raise ValueError(msg)

    # Initialize persistent properties
    self._sigma = sigma
    self._n_actions = n_actions

    # Sample new reward probabilities
    self._new_sess()

  def _new_sess(self):
    # Pick new reward probabilities.
    # Sample randomly between 0 and 1
    self._reward_probs = np.random.rand(self._n_actions)

  def step(self, choice: int) -> int:
    """Run a single trial of the task.

    Args:
      choice: integer specifying choice made by the agent (must be less than
        n_actions.)

    Returns:
      reward: The reward to be given to the agent. 0 or 1.

    """
    # Check inputs
    if choice not in range(self._n_actions):
      msg = (
          f'Found value for choice of {choice}, but must be in '
          f'{list(range(self._n_actions))}')
      raise ValueError(msg)

    # Sample reward with the probability of the chosen side
    reward = np.random.rand() < self._reward_probs[choice]

    # Add gaussian noise to reward probabilities
    drift = np.random.normal(loc=0, scale=self._sigma, size=2)
    self._reward_probs += drift

    # Fix reward probs that've drifted below 0 or above 1
    self._reward_probs = np.maximum(self._reward_probs, [0, 0])
    self._reward_probs = np.minimum(self._reward_probs, [1, 1])

    return reward

  @property
  def reward_probs(self) -> np.ndarray:
    return self._reward_probs.copy()

  @property
  def n_actions(self) -> int:
    return self._n_actions


class BanditSession(NamedTuple):
  """Holds data for a single session of a bandit task."""
  choices: np.ndarray
  rewards: np.ndarray
  timeseries: np.ndarray
  n_trials: int

Agent = Union[AgentQ, AgentNetwork]
Environment = Union[EnvironmentBanditsFlips, EnvironmentBanditsDrift]


def run_experiment(agent: Agent,
                   environment: Environment,
                   n_trials: int) -> BanditSession:
  """Runs a behavioral session from a given agent and environment.

  Args:
    agent: An agent object
    environment: An environment object
    n_trials: The number of steps in the session you'd like to generate

  Returns:
    experiment: A BanditSession holding choices and rewards from the session
  """
  choices = np.zeros(n_trials)
  rewards = np.zeros(n_trials)
  reward_probs = np.zeros((n_trials, 2))

  for trial in np.arange(n_trials):
    # First record environment reward probs
    reward_probs[trial] = environment.reward_probs
    # First agent makes a choice
    choice = agent.get_choice()
    # Then environment computes a reward
    reward = environment.step(choice)
    # Finally agent learns
    agent.update(choice, reward)
    # Log choice and reward
    choices[trial] = choice
    rewards[trial] = reward

  experiment = BanditSession(n_trials=n_trials,
                             choices=choices,
                             rewards=rewards,
                             timeseries=reward_probs)
  return experiment


def plot_session(choices: np.ndarray,
                 rewards: np.ndarray,
                 timeseries: np.ndarray,
                 timeseries_name: str):
  """Plot data from a single behavioral session of the bandit task."""

  choose_high = choices == 1
  choose_low = choices == 0
  rewarded = rewards == 1

  y_high = np.max(timeseries) + 0.1
  y_low = np.min(timeseries) - 0.1

  # Make the plot
  plt.subplots(figsize=(10, 3))
  plt.plot(timeseries)

  # Rewarded high
  plt.scatter(
      np.argwhere(choose_high & rewarded),
      y_high * np.ones(np.sum(choose_high & rewarded)),
      color='green',
      marker=3)
  plt.scatter(
      np.argwhere(choose_high & rewarded),
      y_high * np.ones(np.sum(choose_high & rewarded)),
      color='green',
      marker='|')
  # Omission high
  plt.scatter(
      np.argwhere(choose_high & 1 - rewarded),
      y_high * np.ones(np.sum(choose_high & 1 - rewarded)),
      color='red',
      marker='|')

  # Rewarded low
  plt.scatter(
      np.argwhere(choose_low & rewarded),
      y_low * np.ones(np.sum(choose_low & rewarded)),
      color='green',
      marker='|')
  plt.scatter(
      np.argwhere(choose_low & rewarded),
      y_low * np.ones(np.sum(choose_low & rewarded)),
      color='green',
      marker=2)
  # Omission Low
  plt.scatter(
      np.argwhere(choose_low & 1 - rewarded),
      y_low * np.ones(np.sum(choose_low & 1 - rewarded)),
      color='red',
      marker='|')

  plt.xlabel('Trial')
  plt.ylabel(timeseries_name)


def create_dataset(agent: Agent,
                   environment: Environment,
                   n_trials_per_session: int,
                   n_sessions: int,
                   batch_size: Optional[int] = None):
  """Generates a behavioral dataset from a given agent and environment.

  Args:
    agent: An agent object to generate choices
    environment: An environment object to generate rewards
    n_trials_per_session: The number of trials in each behavioral session to
      be generated
    n_sessions: The number of sessions to generate
    batch_size: The size of the batches to serve from the dataset. If None, 
      batch_size defaults to n_sessions

  Returns:
    A DatasetRNN object suitable for training RNNs.
  """
  xs = np.zeros((n_trials_per_session, n_sessions, 2))
  ys = np.zeros((n_trials_per_session, n_sessions, 1))
  experiment_list = []

  for sess_i in np.arange(n_sessions):
    experiment = run_experiment(agent, environment, n_trials_per_session)
    experiment_list.append(experiment)
    prev_choices = np.concatenate(([0], experiment.choices[0:-1]))
    prev_rewards = np.concatenate(([0], experiment.rewards[0:-1]))
    xs[:, sess_i] = np.swapaxes(
        np.concatenate(([prev_choices], [prev_rewards]), axis=0), 0, 1)
    ys[:, sess_i] = np.expand_dims(experiment.choices, 1)

  dataset = DatasetRNN(xs, ys, batch_size)
  return dataset, experiment_list


###############
# DIAGNOSTICS #
###############


def show_valuemetric(experiment_list):
  """Plot value metric over time from data in experiment_list."""

  reward_prob_bins = np.linspace(-1, 1, 50)
  n_left = np.zeros(len(reward_prob_bins)-1)
  n_right = np.zeros(len(reward_prob_bins)-1)

  for sessdata in experiment_list:
    reward_prob_diff = sessdata.timeseries[:, 0] - sessdata.timeseries[:, 1]
    for reward_prob_i in range(len(n_left)):
      trials_in_bin = np.logical_and(
          (reward_prob_bins[reward_prob_i] < reward_prob_diff) ,
          (reward_prob_diff < reward_prob_bins[reward_prob_i+1]))
      n_left[reward_prob_i] += np.sum(
          np.logical_and(trials_in_bin, sessdata.choices == 0.))
      n_right[reward_prob_i] += np.sum(
          np.logical_and(trials_in_bin, sessdata.choices == 1.))

  choice_probs = n_left / (n_left + n_right)

  xs = reward_prob_bins[:-1] - (reward_prob_bins[1]-reward_prob_bins[0])/2
  ys = choice_probs
  plt.plot(xs, ys)
  plt.ylim((0, 1))
  plt.xlabel('Difference in Reward Probability (left - right)')
  plt.ylabel('Proportion of Leftward Choices')


def show_total_reward_rate(experiment_list):
  rewards = 0
  trials = 0

  for sessdata in experiment_list:
    rewards += np.sum(sessdata.rewards)
    trials += sessdata.n_trials

  print('Total Reward Rate is:', 100*rewards/trials, '%')


################################
# FITTING FUNCTIONS FOR AGENTS #
################################


class HkAgentQ(hk.RNNCore):
  """Vanilla Q-Learning model, expressed in Haiku.

  Updates value of chosen action using a delta rule with step-size param alpha. 
  Does not update value of the unchosen action.
  Selects actions using a softmax decision rule with parameter Beta.
  """

  def __init__(self, n_cs=4):
    super(HkAgentQ, self).__init__()

    # Haiku parameters
    alpha_unsigmoid = hk.get_parameter(
        'alpha_unsigmoid', (1,),
        init=hk.initializers.RandomUniform(minval=-1, maxval=1),
    )
    beta = hk.get_parameter(
        'beta', (1,), init=hk.initializers.RandomUniform(minval=0, maxval=2)
    )

    # Local parameters
    self.alpha = jax.nn.sigmoid(alpha_unsigmoid)
    self.beta = beta
    self._q_init = 0.5

  def __call__(self, inputs: jnp.array, prev_state: jnp.array):
    prev_qs = prev_state

    choice = inputs[:, 0]  # shape: (batch_size, 1)
    reward = inputs[:, 1]  # shape: (batch_size, 1)

    choice_onehot = jax.nn.one_hot(choice, num_classes=2)  # shape: (batch_size, 2)
    chosen_value = jnp.sum(prev_qs * choice_onehot, axis=1)  # shape: (batch_size)
    deltas = reward - chosen_value  # shape: (batch_size)
    new_qs = prev_qs + self.alpha * choice_onehot * jnp.expand_dims(deltas, -1)

    # Compute output logits
    choice_logits = self.beta * new_qs

    return choice_logits, new_qs

  def initial_state(self, batch_size):
    values = self._q_init * jnp.ones([batch_size, 2])  # shape: (batch_size, n_actions)
    return values
