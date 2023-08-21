"""Classes for RL Agents."""
import numpy as np


def check_in_0_1_range(x, name):
  if not (0 <= x <= 1):
    raise ValueError(
        f'Value of {name} must be in [0, 1] range. Found value of {x}.')


def softmax(x: np.ndarray, beta: float = 1) -> np.ndarray:
  """Compute softmax exp(b*x) / sum(exp(b*x)).

  Args:
    x: array containing elements for which to compute softmax
    beta: scalar inverse temperature parameter (default = 1)

  Returns:
    softmax of x.
  """
  return np.exp(beta*x) / np.sum(np.exp(beta*x))


def one_hot(i, n):
  """Convert integer label i to a one-hot of length n."""
  return np.eye(n)[i]


class QAgentABFP:
  """An agent implementing textbook Q-learning, using a softmax policy.
  
  Note: This agent assumes the environment is an n-armed bandit: that is, the 
    environment contains only one state and there are no sequential decisions.
  """

  def __init__(
      self,
      learning_rate: float,
      beta_softmax: float,
      n_actions: int = 2,
      q_init: float = 0.5,
      forgetting_param: float = 0.,
      perseveration_param: float = 0.,
  ):
    """Initialize.

    Args:
      learning_rate: scalar learning rate
      beta_softmax: scalar softmax inverse temperature parameter.
      n_actions: number of actions (default=2)
      q_init: initial value (default=0.5).
      forgetting_param: how much q-values will decay toward the q_init on each
        update step (default=0)
      perseveration_param: how much q-values will decay toward the q_init on
        each update step (default=0)
    """
    self._learning_rate = learning_rate
    self._beta_softmax = beta_softmax
    self._n_actions = n_actions
    self._q_init = q_init
    self._forgetting_param = forgetting_param
    self._perseveration_param = perseveration_param

    # Initialize q-values.
    self._q = q_init * np.ones(n_actions)

    # Check parameter values are in allowed range.
    check_in_0_1_range(learning_rate, 'learning_rate')
    check_in_0_1_range(forgetting_param, 'forgetting_param')
    check_in_0_1_range(perseveration_param, 'perseveration_param')

  def select_action(self): 
    """Select action by sampling from softmax over q-values."""   
    # Compute choice probabilities according to the softmax.
    choice_probs = softmax(self.q, self._beta_softmax)
    # Select a choice according to the probabilities
    chosen_action = np.argmax(np.random.multinomial(1, pvals=choice_probs))
    return chosen_action

  def update(self, action: int, reward: float):
    """Update q values.

    Args:
      action: chosen action
      reward: observed reward
    """
    lr = self._learning_rate
    f = self._forgetting_param
    p = self._perseveration_param

    # Apply forgetting: decay q-values toward initial values.
    self._q = (1 - f) * self._q + f * self._q_init

    # Update the q-value of the chosen action towards the reward.
    self._q[action] = (1 - lr) * self._q[action] + lr * reward

    # Apply perseveration: Update the q-value toward the chosen action.
    self._q = (1 - p) * self._q + p * one_hot(action, self._n_actions)

  @property
  def q(self):
    return self._q.copy()


class QAgent(QAgentABFP):
  """A simple 2-parameter Q-Agent with learning rate and softmax parameter."""

  def __init__(
      self,
      learning_rate: float,
      beta_softmax: float,
      n_actions: int = 2,
      q_init: float = 0.5,
  ):
    """Initialize.

    Args:
      learning_rate: scalar learning rate
      beta_softmax: scalar softmax inverse temperature parameter.
      n_actions: number of actions (default=2)
      q_init: initial value.
    """
    super(QAgent, self).__init__(
        learning_rate=learning_rate,
        beta_softmax=beta_softmax,
        n_actions=n_actions,
        q_init=q_init,
        forgetting_param=0,
        perseveration_param=0,
    )


class QAgent2(QAgentABFP):
  """A more complex Q-Agent with an additional parameter(s)."""

  def __init__(
      self,
      learning_rate: float,
      beta_softmax: float,
      n_actions: int = 2,
      q_init: float = 0.5,
  ):
    """Initialize.

    Args:
      learning_rate: scalar learning rate
      beta_softmax: scalar softmax inverse temperature parameter.
      n_actions: number of actions (default=2)
      q_init: initial value.
    """
    super(QAgent2, self).__init__(
        learning_rate=learning_rate,
        beta_softmax=beta_softmax,
        n_actions=n_actions,
        q_init=q_init,
        forgetting_param=1.,  # forgetting parameter turned on
        perseveration_param=0,
    )


class QAgent3(QAgentABFP):
  """Another more complex Q-Agent with an additional parameter(s)."""

  def __init__(
      self,
      learning_rate: float,
      beta_softmax: float,
      n_actions: int = 2,
      q_init: float = 0.5,
  ):
    """Initialize.

    Args:
      learning_rate: scalar learning rate
      beta_softmax: scalar softmax inverse temperature parameter.
      n_actions: number of actions (default=2)
      q_init: initial value.
    """
    super(QAgent3, self).__init__(
        learning_rate=learning_rate,
        beta_softmax=beta_softmax,
        n_actions=n_actions,
        q_init=q_init,
        forgetting_param=.1,  # forgetting parameter turned on
        perseveration_param=.1,  # perseveration parameter turned on
    )
