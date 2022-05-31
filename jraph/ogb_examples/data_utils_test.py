# Copyright 2020 DeepMind Technologies Limited.


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for graph.ogb_examples.data_utils."""

import pathlib
from absl.testing import absltest
from absl.testing import parameterized
import jraph
from jraph.ogb_examples import data_utils
import numpy as np
import tree


class DataUtilsTest(parameterized.TestCase):

  def setUp(self):
    super(DataUtilsTest, self).setUp()
    self._test_graph = jraph.GraphsTuple(
        nodes=np.broadcast_to(
            np.arange(10, dtype=np.float32)[:, None], (10, 10)),
        edges=np.concatenate((
            np.broadcast_to(np.arange(20, dtype=np.float32)[:, None], (20, 4)),
            np.broadcast_to(np.arange(20, dtype=np.float32)[:, None], (20, 4))
            )),
        receivers=np.concatenate((np.arange(20), np.arange(20))),
        senders=np.concatenate((np.arange(20), np.arange(20))),
        globals={'label': np.array([1], dtype=np.int32)},
        n_node=np.array([10], dtype=np.int32),
        n_edge=np.array([40], dtype=np.int32))
    ogb_path = pathlib.Path(data_utils.__file__).parents[0]
    master_csv_path = pathlib.Path(ogb_path, 'test_data', 'master.csv')
    split_path = pathlib.Path(ogb_path, 'test_data', 'train.csv.gz')
    data_path = master_csv_path.parents[0]
    self._reader = data_utils.DataReader(
        data_path=data_path,
        master_csv_path=master_csv_path,
        split_path=split_path)

  def test_total_num_graph(self):
    self.assertEqual(self._reader.total_num_graphs, 1)

  def test_expected_graph(self):
    graph = next(self._reader)
    with self.subTest('test_graph_equality'):
      tree.map_structure(
          np.testing.assert_almost_equal, graph, self._test_graph)
    with self.subTest('stop_iteration'):
      # One element in the dataset, so should have stop iteration.
      with self.assertRaises(StopIteration):
        next(self._reader)

  def test_reader_repeat(self):
    self._reader.repeat()
    next(self._reader)
    graph = next(self._reader)
    # One graph in the test dataset so should be the same.
    tree.map_structure(np.testing.assert_almost_equal, graph, self._test_graph)


if __name__ == '__main__':
  absltest.main()
