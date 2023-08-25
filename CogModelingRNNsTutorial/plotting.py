"""Plotting code."""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


def confidence_interval(data, alpha=0.95):
    return st.t.interval(
        alpha=alpha, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))


def action_similarity_to_history(experiment_list, n):
  """Compute rate at which each action equals the action i steps ago for i in (1, 2, ..., n)."""
  lags = np.zeros((n-1, len(experiment_list)))
  ci95 = np.zeros((n-1, 2))
  for k in range(1, n):
    for i, expt in enumerate(experiment_list):
      lags[k-1, i] += np.mean(expt.choices[:-k] == expt.choices[k:])
    ci95[k-1] = confidence_interval(lags[k-1])
  return np.mean(lags, axis=1), ci95


def plot_action_similarity_to_history(*experiment_lists, n_steps_back=16, labels=None, ax=None, **legend_kwargs):
  """Plot rate at which each action equals the action i steps ago for i in (1, 2, ..., n).

  Args:
    experiment_lists: experiment lists to evaluate + plot
    n_steps_back: number of steps to go back
    labels: If provided, labels for each experiment
    ax: plotting axes (optional)
  """
  do_legend = True 
  if labels is None:
    do_legend = False
    labels = [None] * len(experiment_lists)

  if ax is None:
    ax = plt.gca()

  for i, expt in enumerate(experiment_lists):
    if expt is not None:
      lag, ci95 = action_similarity_to_history(expt, n_steps_back)
      ax.plot(np.arange(1, n_steps_back), lag, label=labels[i])
      ax.fill_between(np.arange(1, n_steps_back), ci95[:, 0], ci95[:, 1], alpha=0.25)

    if do_legend:
      ax.legend(bbox_to_anchor=(1, 1))
    ax.set_ylabel('Choice Similarity')
    ax.set_xlabel('Number of steps in past')
