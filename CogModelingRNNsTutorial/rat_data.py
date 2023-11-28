"""Functions for loading rat data."""
import json
import os
import numpy as np
import pickle

from typing import List, Optional


JSON_PATH = "./CogModelingRNNsTutorial/data/miller2018_all_rats.json"  # Where raw rat data is stored
DATA_DIR = "./CogModelingRNNsTutorial/data/rat_data/"  # Where you will save out individual rat data
PREFIX = "miller2018"


def _get_single_rat_fname(rat_id):
  assert isinstance(rat_id, int)
  rat_id_padded = f'{rat_id}'.rjust(2, '0')
  return f"{PREFIX}_rat{rat_id_padded}.npy"


def load_data_for_one_rat(fname=None, data_dir=DATA_DIR):
  """Load data for a single rat.

  Args:
    fname: name of file (will likely be the name of a npy file you loaded
    data_dir: directory where file lives


  Returns:
    xs: n_trials x n_sessions x 2 array of choices and rewards
    ys: n_trials x n_sessions x 1 array of choices (shifted forward
      by one compared to xs[..., 0]).
    fname: name of file
  """
  if not os.path.exists(data_dir):
    raise ValueError(f'data_dir {data_dir} not found.')

  if fname is None:
    rat_files = [f for f in os.listdir(data_dir) if (f.startswith(f'{PREFIX}_rat') and f.endswith('.npy'))]
    fname = rat_files[np.random.randint(len(rat_files))]
    print(f'Loading data from {fname}.')
  else:
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
      raise ValueError(f'path {fpath} not found.')

  data = np.load(os.path.join(data_dir, fname))
  xs, ys = data[..., :2], data[..., 2:]
  assert ys.shape[-1] == 1
  assert ys.ndim == xs.ndim == 3
  return xs, ys, fname


def format_into_datasets(xs, ys, dataset_constructor):
  """Format inputs xs and outputs ys into dataset.

  Args:
    xs: n_trials x n_sessions x 2 array of choices and rewards
    ys: n_trials x n_sessions x 1 array of next choice. choice value of -1 denotes
      instructed trial or padding at end of session.
    dataset_constructor: constructor that accepts xs and ys as arguments; probably
      use rnn_utils.DatasetRNN

  Returns:
    dataset_train: a dataset containing even numbered sessions
    dataset_train: a dataset containing odd numbered sessions
  """
  n = int(xs.shape[1] // 2) * 2
  dataset_train = dataset_constructor(xs[:, :n:2], ys[:, :n:2])
  dataset_test = dataset_constructor(xs[:, 1:n:2], ys[:, 1:n:2])
  return dataset_train, dataset_test


def find(s, ch):
  """Find index of character within string."""
  return [i for i, ltr in enumerate(s) if ltr == ch]


def get_rat_bandit_datasets(data_file: Optional[str] = None):
  """Packages up the rat datasets.

  Requires downloading the dataset file "tab_dataset.json", which is available
  on Figshare.

  https://figshare.com/articles/dataset/From_predictive_models_to_cognitive_models_Separable_behavioral_processes_underlying_reward_learning_in_the_rat/20449356

  Args:
    data_file: Complete path to the dataset file, including the filename. If not
      specified, will look for data in the predictive_cognitive folder on CNS.

  Returns:
    A list of DatasetRNN objects. One element per rat.
    In each of these, each session will be an episode, padded with NaNs
    to match length. "ys" will be the choices on each trial
    (left=0, right=1) "xs" will be the choice and reward (0 or 1) from
    the previous trial. Invalid xs and ys will be -1
  """
  if data_file is None:
    data_file = '/cns/ej-d/home/kevinjmiller/predictive_cognitive/tab_dataset.json'

  with open(data_file, 'r') as f:
    dataset = json.load(f)

  n_rats = len(dataset)

  dataset_list = []
  # Each creates a DatasetRNN object for a single rat, adds it to the list
  for rat_i in range(n_rats):
    ratdata = dataset[rat_i]
    sides = ratdata['sides']
    n_trials = len(sides)

    # Left choices will be 0s, right choices will be 1s, viols will be removed
    rights = find(sides, 'r')
    choices = np.zeros(n_trials)
    choices[rights] = 1

    vs = find(sides, 'v')
    viols = np.zeros(n_trials, dtype=bool)
    viols[vs] = True

    # Free will be 0 and forced will be 1
    free = find(ratdata['trial_types'], 'f')
    instructed_choice = np.ones(n_trials)
    instructed_choice[free] = 0

    rewards = np.array(ratdata['rewards'])
    new_sess = np.array(ratdata['new_sess'])

    n_sess = np.sum(new_sess)
    sess_starts = np.nonzero(np.concatenate((new_sess, [1])))[0]
    max_session_length = np.max(np.diff(sess_starts, axis=0))

    # Populate matrices for rewards and choices. size (n_trials, n_sessions, 1)
    rewards_by_session = -1 * np.ones((max_session_length, n_sess, 1))
    choices_by_session = -1 * np.ones((max_session_length, n_sess, 1))
    instructed_by_session = -1 * np.ones((max_session_length, n_sess, 1))

    # Each iteration processes one session
    for sess_i in np.arange(n_sess):
      sess_start = sess_starts[sess_i]
      sess_end = sess_starts[sess_i + 1]

      viols_sess = viols[sess_start:sess_end]
      rewards_sess = rewards[sess_start:sess_end]
      choices_sess = choices[sess_start:sess_end]
      instructed_choice_sess = instructed_choice[sess_start:sess_end]

      rewards_sess = np.delete(rewards_sess, viols_sess)
      choices_sess = np.delete(choices_sess, viols_sess)
      instructed_choice_sess = np.delete(instructed_choice_sess, viols_sess)

      sess_length_noviols = len(choices_sess)

      rewards_by_session[0:sess_length_noviols, sess_i, 0] = rewards_sess
      choices_by_session[0:sess_length_noviols, sess_i, 0] = choices_sess
      instructed_by_session[0:sess_length_noviols, sess_i, 0] = (
          instructed_choice_sess
      )

    # Inputs: choices and rewards, offset by one trial
    choice_and_reward = np.concatenate(
        (choices_by_session, rewards_by_session), axis=2
    )
    # Add a dummy input at the beginning. First step has a target but no input
    xs = np.concatenate(
        (0. * np.ones((1, n_sess, 2)), choice_and_reward), axis=0
    )
    # Targets: choices on each free-choice trial
    free_choices = choices_by_session
    free_choices[instructed_by_session == 1] = -1
    # Add a dummy target at the end -- last step has input but no target
    ys = np.concatenate((free_choices, -1*np.ones((1, n_sess, 1))), axis=0)

    dataset_list.append([xs, ys])

  return dataset_list


def save_out_rat_data_as_pickle(json_path=JSON_PATH, data_dir=DATA_DIR, verbose=True):
  """Load json with all rat data + save out individual RNNDatasets for each rat."""
  if not os.path.exists(json_path):
    raise ValueError(f'json_path {json_path} does not exist.')

  # Make destination directory if it does not already exists.
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  if verbose:
    print(f'Loading data from {json_path}.')
  dataset = get_rat_bandit_datasets(json_path)

  if verbose:
    print(f'Saving out data to {data_dir}.')

  for rat_id in range(len(dataset)):
    fname = _get_single_rat_fname(rat_id)
    save_path = os.path.join(data_dir, fname)
    xs, ys = dataset[rat_id]
    np.save(save_path, np.concatenate([xs, ys], axis=-1))

