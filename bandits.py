from typing import Callable, NamedTuple, Tuple, Union, Optional

import numpy as np
import haiku as hk
import matplotlib.pyplot as plt
import jax

from CogModelingRNNsTutorial.rnn_utils import DatasetRNN

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
      alpha: float,  # Learning rate
      beta: float,  # softmax temp
  ):
    self._alpha = alpha
    self._beta = beta
    self.new_sess()

  def new_sess(self):
    """Reset the agent for the beginning of a new session."""
    self.q = 0.5 * np.ones(2)

  def get_choice_probs(self) -> np.ndarray:
    choice_probs = np.exp(self._beta * self.q) / np.sum(
        np.exp(self._beta * self.q))
    return choice_probs

  def get_choice(self) -> int:
    """Sample a choice, given the agent's current internal state."""

    choice_probs = self.get_choice_probs()
    choice = np.random.choice(2, p=choice_probs)
    return choice

  def update(self,
             choice: int,
             reward: int):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent. 0 or 1
    """
    self.q[choice] = (1 - self._alpha) * self.q[choice] + self._alpha * reward


class AgentLeakyActorCritic:
  """An agent that runs Actor-Critic learning for the y-maze tasks.

  Attributes:
    alpha_critic: The critic's learning rate
    alpha_actor: The actor's learning rate
    gamma: The actor's forgetting rate
    theta: The actor's policy parameter
    v: The critic's estimate of environmental reward rate

  """

  def __init__(
      self,
      alpha_critic: float,
      alpha_actor_learn: float,
      alpha_actor_forget: float,
  ):
    self._alpha_critic = alpha_critic
    self._alpha_actor_learn = alpha_actor_learn
    self._alpha_actor_forget = alpha_actor_forget
    self.new_sess()

  def new_sess(self):
    """Reset the agent for the beginning of a new session."""
    self.theta = 0. * np.ones(2)
    self.v = 0.5

  def get_choice_probs(self) -> np.ndarray:
    choice_probs = np.exp(self.theta) / np.sum(
        np.exp(self.theta))
    return choice_probs

  def get_choice(self) -> int:
    """Sample a choice, given the agent's current internal state."""

    choice_probs = self.get_choice_probs()
    choice = np.random.choice(2, p=choice_probs)
    return choice

  def update(self, choice: int, reward: int):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent. 0 or 1
    """
    unchosen = 1 - choice  # Convert 0 to 1 or 1 to 0
    # Actor learning
    choice_probs = self.get_choice_probs()
    self.theta[choice] = (1 - self._alpha_actor_forget) * self.theta[
        choice
    ] + self._alpha_actor_learn * (reward - self.v) * (1 - choice_probs[choice])
    self.theta[unchosen] = (1 - self._alpha_actor_forget) * self.theta[
        unchosen
    ] - self._alpha_actor_learn * (reward - self.v) * (choice_probs[unchosen])

    # Critic learing: V moves towards reward
    self.v = (1 - self._alpha_critic) * self.v + self._alpha_critic * reward


class AgentNetwork:
  """A class that allows running a trained RNN as an agent.

  Attributes:
    make_network: A Haiku function that returns an RNN architecture
    params: A set of Haiku parameters suitable for that architecture
  """

  def __init__(self,
               make_network: Callable[[], hk.RNNCore],
               params: hk.Params):

    def step_network(xs: np.ndarray,
                     state: hk.State) -> Tuple[np.ndarray, hk.State]:
      core = make_network()
      y_hat, new_state = core(xs, state)
      return y_hat, new_state

    def get_initial_state() -> hk.State:
      core = make_network()
      state = core.initial_state(1)
      return state

    model = hk.without_apply_rng(hk.transform(step_network))
    state = hk.without_apply_rng(hk.transform(get_initial_state))

    self._initial_state = state.apply(params)
    self._model_fun = jax.jit(lambda xs, state: model.apply(params, xs, state))
    self._xs = np.zeros((1, 2))
    self.new_sess()

  def new_sess(self):
    self._state = self._initial_state

  def get_choice_probs(self) -> np.ndarray:
    output_logits, _ = self._model_fun(self._xs, self._state)
    choice_probs = np.exp(output_logits[0]) / np.sum(
        np.exp(output_logits[0]))
    return choice_probs

  def get_choice(self) -> Tuple[int, np.ndarray]:
    choice_probs = self.get_choice_probs()
    choice = np.random.choice(2, p=choice_probs)
    return choice

  def update(self, choice: int, reward: int):
    self._xs = np.array([[choice, reward]])
    _, self._state = self._model_fun(self._xs, self._state)

################
# ENVIRONMENTS #
################

class EnvironmentBanditsFlips:
  # An environment for a two-armed bandit RL task, with reward probabilities that flip in blocks

  def __init__(
      self,
      block_flip_prob,
      reward_prob_high,
      reward_prob_low,
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
    # Flip the block
    self._block = 1 - self._block
    # Set the reward probabilites
    if self._block == 1:
      self.reward_probabilities = [self._reward_prob_high,
                                   self._reward_prob_low]
    else:
      self.reward_probabilities = [self._reward_prob_low,
                                   self._reward_prob_high]

  def next_trial(self, choice):
    # Choose the reward probability associated with the choice that the agent made
    reward_prob_trial = self.reward_probabilities[choice]

    # Sample a reward with this probability
    reward = np.random.binomial(1, reward_prob_trial)

    # Check whether to flip the block
    if np.random.binomial(1, self._block_flip_prob):
      self.new_block()

    # Return the reward
    return reward


class EnvironmentBanditsDrift:
  """Environment for a drifting two-armed bandit task.

  Reward probabilities on each arm are sampled randomly between 0 and
  1. On each trial, gaussian random noise is added to each.

  Attributes:
    sigma: A float, between 0 and 1, giving the magnitude of the drift
    reward_probs: Probability of reward associated with each action
  """

  def __init__(self,
               sigma: float,
               ):

    # Check inputs
    if sigma < 0:
      msg = ('sigma was {}, but must be greater than 0')
      raise ValueError(msg.format(sigma))
    # Initialize persistent properties
    self._sigma = sigma

    # Sample new reward probabilities
    self._new_sess()

  def _new_sess(self):
    # Pick new reward probabilities.
    # Sample randomly between 0 and 1
    self.reward_probs = np.random.rand(2)

  def step(self,
           choice: int) -> int:
    """Run a single trial of the task.

    Args:
      choice: The choice made by the agent. 0 or 1

    Returns:
      reward: The reward to be given to the agent. 0 or 1.

    """
    # Check inputs
    if not np.logical_or(choice == 0, choice == 1):
      msg = ('choice given was {}, but must be either 0 or 1')
      raise ValueError(msg.format(choice))

    # Sample reward with the probability of the chosen side
    reward = np.random.rand() < self.reward_probs[choice]
    # Add gaussian noise to reward probabilities
    drift = np.random.normal(loc=0, scale=self._sigma, size=2)
    self.reward_probs += drift

    # Fix reward probs that've drifted below 0 or above 1
    self.reward_probs = np.maximum(self.reward_probs, [0, 0])
    self.reward_probs = np.minimum(self.reward_probs, [1, 1])

    return reward


class BanditSession(NamedTuple):
  """Holds data for a single session of a bandit task."""
  choices: np.ndarray
  rewards: np.ndarray
  timeseries: np.ndarray
  n_trials: int

Agent = Union[AgentQ, AgentLeakyActorCritic, AgentNetwork]
Environment = Union[
    EnvironmentBanditsFlips, EnvironmentBanditsDrift
]

def run_experiment(agent: Agent,
                   environment: Environment,
                   n_trials: int) -> BanditSession:
  """Runs a behavioral session from a given agent and environment.

  Args:
    agent: An agent object
    environment: An environment object
    n_steps: The number of steps in the session you'd like to generate

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
                    n_trials: int,
                    timeseries: np.ndarray,
                timeseries_name: str):
  """Creates a figure showing data from a single behavioral session of the bandit task.
  """

  choose_high = choices == 1
  choose_low = choices == 0
  rewarded = rewards == 1

  # Make the plot
  plt.subplots(figsize=(10, 3))
  plt.plot(timeseries)

  # Rewarded high
  plt.scatter(
      np.argwhere(choose_high & rewarded),
      1.1 * np.ones(np.sum(choose_high & rewarded)),
      color='green',
      marker=3)
  plt.scatter(
      np.argwhere(choose_high & rewarded),
      1.1 * np.ones(np.sum(choose_high & rewarded)),
      color='green',
      marker='|')
  # Omission high
  plt.scatter(
      np.argwhere(choose_high & 1 - rewarded),
      1.1 * np.ones(np.sum(choose_high & 1 - rewarded)),
      color='red',
      marker='|')

  # Rewarded low
  plt.scatter(
      np.argwhere(choose_low & rewarded),
      -0.1 * np.ones(np.sum(choose_low & rewarded)),
      color='green',
      marker='|')
  plt.scatter(
      np.argwhere(choose_low & rewarded),
      -0.1 * np.ones(np.sum(choose_low & rewarded)),
      color='green',
      marker=2)
  # Omission Low
  plt.scatter(
      np.argwhere(choose_low & 1 - rewarded),
      -0.1 * np.ones(np.sum(choose_low & 1 - rewarded)),
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


################################
# FITTING FUNCTIONS FOR AGENTS #
################################
