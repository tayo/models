# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Coerce internally generated data into the format expected by NCF."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import os
import pickle
import re
import sys
import time

# pylint: disable=g-bad-import-order
import numpy as np
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.recommendation import constants as rconst

FLAGS = flags.FLAGS

flags.DEFINE_string(
    name="raw_data_pattern",
    default="gs://tayo_datasets/ml20m_extended/ncf_data_4_16_",
    help="Matching pattern for sharded test and train files. The string "
    "supplied will be run against gfile.Glob(). Shards with 'train' will be "
    "used for training in alphabetical order. Likewise for 'test'. "
    "The path should not include the substring 'metadata'.")

flags.DEFINE_string(
    name="output_dir",
    default="/tmp/",
    help="Directory for storing output custom cache.")

flags.DEFINE_string(
    name="output_fn",
    default="transformed.pkl",
    help="Filename for the output pkl.")

flags.DEFINE_integer(
    name="max_positive_count",
    default=int(1e12),
    help="Maximum number of elements to consider.")


def pkl_iterator_gen(path, max_count):
  """Pickle iterator function using a generator.

    Note: Some older packing formats of this dataset return a list of numpy
    objects directly, instead of list of lists. For those, use the following:
      for user_id, items in enumerate(pkl_iterator(..))
  """

  pkl_options = {}
  if sys.version_info[0] >= 3:
    pkl_options["encoding"] = "bytes"

  with tf.gfile.Open(path, "rb") as f:
    count = 0
    user_id = -1
    while True:
      try:
        x = pickle.load(f, **pkl_options)
        count += len(x)
        user_id += 1
        if count >= max_count:
          break

        if not (user_id + 1) % 10000:
          print(str(user_id + 1).ljust(10), "{:.2E}".format(count))

        yield x

      except EOFError:
        break


def pkl_data(path):

  pkl_options = {}
  if sys.version_info[0] >= 3:
    pkl_options["encoding"] = "bytes"

  with tf.gfile.Open(path, "rb") as f:
    try:
      pdata = pickle.load(f, **pkl_options)
      return pdata

    except EOFError:
      print("EOFError during pickle.load()")


def process_train_files(train_files):
  """Process the train files.

  Args:
    train_files: Array of data files to process sequentially.

  Returns:
    num_users: An integer
    train_users: An array
    train_items: An array
    item_map: A collections.defaultdict
  """
  print("Train files: Starting first pass")
  num_users = 0
  item_counts = collections.defaultdict(int)
  start_ind = 0
  for tf in train_files:
    print("  Processing train file {}".format(tf))
    elems = pkl_data(tf)
    assert isinstance(elems, list)
    assert type(elems[0])==np.ndarray

    for user_id, item_list in enumerate(elems):
      for i in item_list:
        item_counts[i] += 1
    num_users = num_users + len(elems)
      
  # Sort items by popularity to increase the efficiency of the bisection lookup.
  item_map = sorted([(v, k) for k, v in item_counts.items()], reverse=True)
  item_map = {j: i for i, (_, j) in enumerate(item_map)}

  print("Computing dataset statistics.")
  num_positives = sum(item_counts.values())
  num_items = len(item_map)
  print("  num_pts: {}, num_users: {}, num_items: {}".format(
        num_positives, num_users, num_items))

  assert num_users <= np.iinfo(rconst.USER_DTYPE).max
  assert num_items <= np.iinfo(rconst.ITEM_DTYPE).max
  train_users = np.zeros(shape=num_positives, dtype=rconst.USER_DTYPE) - 1
  train_items = np.zeros(shape=num_positives, dtype=rconst.ITEM_DTYPE) - 1

  print("Train files: Starting second pass")
  start_ind = 0
  for tf in train_files:
    print("  Processing train file {}".format(tf))
    elems = pkl_data(tf)
    assert isinstance(elems, list)
    assert type(elems[0])==np.ndarray
    for user_id, item_list in enumerate(elems):
      items = [item_map[i] for i in item_list]
      train_users[start_ind:start_ind + len(items)] = user_id
      train_items[start_ind:start_ind + len(items)] = np.array(
          items, dtype=rconst.ITEM_DTYPE)
      start_ind += len(items)

  assert start_ind == num_positives
  assert not np.any(train_users == -1)
  assert not np.any(train_items == -1)

  return num_users, train_users, train_items, item_map


def process_test_files(test_files, item_map, num_users):
  """Process the test files.

  Args:
    test_files: Array of data files to process sequentially.

  Returns:
    eval_items: An array
  """
  print("Test files: Starting pass")
  np.random.seed(0)
  eval_items = np.zeros(shape=num_users, dtype=rconst.ITEM_DTYPE) - 1
  spare_item = len(item_map) + 1
  start_ind = 0

  for tf in test_files:
    print("  Processing test file {}".format(tf))
    elems = pkl_data(tf)
    assert isinstance(elems, list)
    assert type(elems[0])==np.ndarray
    for user_id, item_list in enumerate(elems):
      items = []
      # Some eval items were not in the training set. Since the item_map encodes
      # popularity, we assign missing items with a unique item representing very
      # low popularity (missing from the training set).
      for i in item_list:
        if i in item_map:
          items.append(item_map[i])
        else:
          items.append(spare_item)
          spare_item += 1
      # Randomly choose an item to add to the eval set.
      np.random.shuffle(items)
      eval_items[start_ind+user_id] = items.pop()
    start_ind += len(elems)

  assert start_ind == num_users
  assert not np.any(eval_items == -1)

  return eval_items


def main(_):
  if "metadata" in FLAGS.raw_data_pattern:
    print("raw_data_pattern should not include the substring 'metadata'.")
    return

  train_files = tf.gfile.Glob(FLAGS.raw_data_pattern + "train*")
  test_files = tf.gfile.Glob(FLAGS.raw_data_pattern + "test*")
  # Do not include the metadata files.
  train_files = [fn for fn in train_files if not "metadata" in fn]
  test_files = [fn for fn in test_files if not "metadata" in fn]

  def alpha_num(s):
    """Split a string, turning numbers into ints to enable natural sort."""
    return [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", s)]
  # Choose natural sort for testing and training filenames to ensure that they
  # are parsed in identical order.
  train_files = sorted(train_files, key=alpha_num)
  test_files = sorted(test_files, key=alpha_num)

  assert len(train_files) > 0
  assert len(test_files) > 0

  print("{} train file(s) found: {}".format(len(train_files), train_files))
  print("{} test file(s) found: {}".format(len(test_files), test_files))

  num_users, train_users, train_items, item_map = process_train_files(
      train_files)
  eval_users = np.arange(num_users, dtype=rconst.USER_DTYPE)
  eval_items = process_test_files(test_files, item_map, num_users)

  data = {
    rconst.TRAIN_USER_KEY: train_users,
    rconst.TRAIN_ITEM_KEY: train_items,
    rconst.EVAL_USER_KEY: eval_users,
    rconst.EVAL_ITEM_KEY: eval_items,
    rconst.USER_MAP: {i for i in range(num_users)},
    rconst.ITEM_MAP: item_map,
    "create_time": time.time() + int(1e10),  # never invalidate.
  }

  print("Writing record.")
  output_path = os.path.join(FLAGS.output_dir, FLAGS.output_fn)
  with tf.gfile.Open(output_path, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
  absl_app.run(main)

