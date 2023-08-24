"""Functions for loading rat data."""
import json
import os
import numpy as np
import pickle

from typing import List, Optional
from . import rnn_utils


JSON_PATH = "./CogModelingRNNsTutorial/data/miller2019_all_mice.json"  # Where raw mouse data is stored
PICKLE_DIR = "./CogModelingRNNsTutorial/data/pickle_files/"  # Where you will save out individual mouse data


def _get_pickle_fname(mouse_id):
  assert isinstance(mouse_id, int)
  mouse_id_padded = f'{mouse_id}'.rjust(2, '0')
  return f"miller2019_mouse{mouse_id_padded}.pickle"


def load_data_for_one_mouse(fname=None, pickle_dir=PICKLE_DIR, mouse_id=None):
  """Load data for a single mouse."""
  if not os.path.exists(pickle_dir):
    raise ValueError(f'pickle_dir {pickle_dir} not found.')

  if fname is None:
    print('a')
    mouse_files = [f for f in os.listdir(pickle_dir) if (f.startswith('miller2019_mouse') and f.endswith('.pickle'))]
    if mouse_id is None:  # Select a random mouse from those available.
      fname = mouse_files[np.random.randint(len(mouse_files))]
      print(f'Loading data from {fname}.')

    else:
      print('b')
      fname = _get_pickle_fname(mouse_id)
      if fname not in mouse_files:
        raise ValueError((
            f'File {fname} not found in {pickle_dir}; found {mouse_files}. '
            'Check mouse_id and pickle_dir are correct.'))
  else:
    print('c')
    fpath = os.path.join(pickle_dir, fname)
    if not os.path.exists(fpath):
      raise ValueError(f'path {fpath} not found.')

  with open(os.path.join(pickle_dir, fname), 'rb') as f:
    data = pickle.load(f)
  return data, fname


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

    dataset_rat = rnn_utils.DatasetRNN(ys=ys, xs=xs)
    dataset_list.append(dataset_rat)

  return dataset_list


def save_out_mouse_data_as_pickle(json_path=JSON_PATH, pickle_dir=PICKLE_DIR, verbose=True):
  """Load json with all mouse data + save out individual RNNDatasets for each mouse."""
  if not os.path.exists(json_path):
    raise ValueError(f'json_path {json_path} does not exist.')

  # Make destination directory if it does not already exists.
  if not os.path.exists(pickle_dir):
    os.makedirs(pickle_dir)

  if verbose:
    print(f'Loading data from {json_path}.')
  dataset = get_rat_bandit_datasets(json_path)

  if verbose:
    print(f'Saving out data to {pickle_dir}.')

  for mouse_id in range(len(dataset)):
    fname = _get_pickle_fname(mouse_id)
    save_path = os.path.join(pickle_dir, fname)

    with open(save_path, 'wb') as f:
      pickle.dump(dataset[mouse_id], f)

