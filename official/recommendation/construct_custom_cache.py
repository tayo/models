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
#import cPickle as pickle
import pickle
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
    name="raw_data_path",
    default="/tmp/ml_extended/16_32_ext_user_item_sequences.pkl",
    help="Path to data positives.")

flags.DEFINE_string(
    name="output_dir",
    default="/tmp/",
    help="Directory for storing output custom cache.")

flags.DEFINE_integer(
    name="max_positive_count",
    default=int(1e12),
    help="Maximum number of elements to consider.")



def pkl_iterator(path, max_count):
  with tf.gfile.Open(path, "rb") as f:
    count = 0
    user_id = -1
    while True:
      try:
        x = pickle.load(f)
        count += len(x)
        user_id += 1
        if count >= max_count:
          break

        if not (user_id + 1) % 10000:
          print(str(user_id + 1).ljust(10), "{:.2E}".format(count))

        yield x

        if count >= max_count:
          break

      except EOFError:
        break


def main(_):
  item_counts = collections.defaultdict(int)
  user_id = -1
  print("Starting precompute pass.")
  for user_id, items in enumerate(
      pkl_iterator(
        FLAGS.raw_data_path,
        FLAGS.max_positive_count)):
    for i in items:
      item_counts[i] += 1

  print("Computing dataset statistics.")
  num_positives = sum(item_counts.values())

  # Sort items by popularity to increase the efficiency of the bisection lookup
  item_map = sorted([(v, k) for k, v in item_counts.items()], reverse=True)
  item_map = {j: i for i, (_, j) in enumerate(item_map)}

  num_users = user_id + 1
  num_items = len(item_map)

  print("num_pts:  ", num_positives)
  print("num_users:", num_users)
  print("num_items:", num_items)

  assert num_users <= np.iinfo(rconst.USER_DTYPE).max
  assert num_items <= np.iinfo(rconst.ITEM_DTYPE).max

  num_train_pts = num_positives - num_users
  train_users = np.zeros(shape=num_train_pts, dtype=rconst.USER_DTYPE) - 1
  train_items = np.zeros(shape=num_train_pts, dtype=rconst.ITEM_DTYPE) - 1
  eval_users = np.arange(num_users, dtype=rconst.USER_DTYPE)
  eval_items = np.zeros(shape=num_users, dtype=rconst.ITEM_DTYPE) - 1

  np.random.seed(0)
  start_ind = 0
  print("Starting second pass.")
  for user_id, items in enumerate(pkl_iterator(FLAGS.raw_data_path,
        FLAGS.max_positive_count)):
    items = [item_map[i] for i in items]

    # randomly choose an item to be the holdout item
    np.random.shuffle(items)

    eval_items[user_id] = items.pop()

    train_users[start_ind:start_ind + len(items)] = user_id
    train_items[start_ind:start_ind + len(items)] = np.array(
        items, dtype=rconst.ITEM_DTYPE)
    start_ind += len(items)

  assert start_ind == num_train_pts
  assert not np.any(train_users == -1)
  assert not np.any(train_items == -1)
  assert not np.any(eval_items == -1)

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
  output_path = os.path.join(FLAGS.output_dir, "transformed.pkl")
  with tf.gfile.Open(output_path, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
  absl_app.run(main)

