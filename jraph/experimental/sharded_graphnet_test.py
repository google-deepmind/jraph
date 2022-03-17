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
"""Tests for sharded graphnet."""

import functools
import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.lib import xla_bridge
import jax.tree_util as tree
import jraph
from jraph._src import utils
from jraph.experimental import sharded_graphnet
import numpy as np


def _get_graphs_from_n_edge(n_edge):
  """Get a graphs tuple from n_edge."""
  graphs = []
  for el in n_edge:
    graphs.append(
        jraph.GraphsTuple(
            nodes=np.random.uniform(size=(128, 2)),
            edges=np.random.uniform(size=(el, 2)),
            senders=np.random.choice(128, el),
            receivers=np.random.choice(128, el),
            n_edge=np.array([el]),
            n_node=np.array([128]),
            globals=np.array([[el]]),
        ))
  graphs = utils.batch_np(graphs)
  return graphs


def get_graphs_tuples(n_edge, sharded_n_edge, device_graph_idx):
  sharded_n_edge = np.array(sharded_n_edge)
  device_graph_idx = np.array(device_graph_idx)
  devices = len(sharded_n_edge)
  graphs = _get_graphs_from_n_edge(n_edge)
  sharded_senders = np.reshape(graphs.senders, [devices, -1])
  sharded_receivers = np.reshape(graphs.receivers, [devices, -1])
  sharded_edges = np.reshape(graphs.edges,
                             [devices, -1, graphs.edges.shape[-1]])
  # Broadcast replicated features to have a devices leading axis.
  broadcast = lambda x: np.broadcast_to(x[None, :], [devices] + list(x.shape))

  sharded_graphs = sharded_graphnet.ShardedEdgesGraphsTuple(
      device_senders=sharded_senders,
      device_receivers=sharded_receivers,
      device_edges=sharded_edges,
      device_n_edge=sharded_n_edge,
      nodes=broadcast(graphs.nodes),
      senders=broadcast(graphs.senders),
      receivers=broadcast(graphs.receivers),
      device_graph_idx=device_graph_idx,
      globals=broadcast(graphs.globals),
      n_node=broadcast(graphs.n_node),
      n_edge=broadcast(graphs.n_edge))
  return graphs, sharded_graphs


class ShardedGraphnetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    os.environ[
        'XLA_FLAGS'] = '--xla_force_host_platform_device_count=3'
    xla_bridge.get_backend.cache_clear()

  @parameterized.named_parameters(
      ('split_3_to_4', [3, 5, 4], [[3, 3], [2, 4]], [[0, 1], [1, 2]]),
      ('split_zero_last_edge', [1, 2, 5, 4], [[1, 2, 3], [2, 4, 0]
                                             ], [[0, 1, 2], [2, 3, 0]]),
      ('split_one_over_multiple', [1, 11], [[1, 3], [4, 0], [4, 0]
                                           ], [[0, 1], [1, 0], [1, 0]]))
  def test_get_sharded_graphs_tuple(self, n_edge, sharded_n_edge,
                                    device_graph_idx):
    in_tuple, expect_tuple = get_graphs_tuples(n_edge, sharded_n_edge,
                                               device_graph_idx)
    out_tuple = sharded_graphnet.graphs_tuple_to_broadcasted_sharded_graphs_tuple(
        in_tuple, num_shards=len(expect_tuple.nodes))
    tree.tree_multimap(np.testing.assert_almost_equal, out_tuple, expect_tuple)

  @parameterized.named_parameters(
      ('split_intermediate', [3, 5, 4, 3, 3]),
      ('split_zero_last_edge', [1, 2, 5, 4, 6]),
      ('split_one_over_multiple', [1, 11]))
  def test_sharded_same_as_non_sharded(self, n_edge):
    in_tuple = _get_graphs_from_n_edge(n_edge)
    devices = 3
    sharded_tuple = sharded_graphnet.graphs_tuple_to_broadcasted_sharded_graphs_tuple(
        in_tuple, devices)
    update_fn = jraph.concatenated_args(lambda x: x)
    sharded_gn = sharded_graphnet.ShardedEdgesGraphNetwork(
        update_fn, update_fn, update_fn, num_shards=devices)
    gn = jraph.GraphNetwork(update_fn, update_fn, update_fn)
    sharded_out = jax.pmap(sharded_gn, axis_name='i')(sharded_tuple)
    expected_out = gn(in_tuple)
    reduced_out = sharded_graphnet.broadcasted_sharded_graphs_tuple_to_graphs_tuple(
        sharded_out)
    jax.tree_util.tree_map(
        functools.partial(np.testing.assert_allclose, atol=1E-5, rtol=1E-5),
        expected_out, reduced_out)


if __name__ == '__main__':
  absltest.main()
