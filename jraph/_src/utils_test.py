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
"""Tests for jraph.utils."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import test_util
import jax.numpy as jnp
import jax.tree_util as tree
from jraph._src import graph
from jraph._src import utils
import numpy as np


def _get_random_graph(max_n_graph=10,
                      include_node_features=True,
                      include_edge_features=True,
                      include_globals=True):
  n_graph = np.random.randint(1, max_n_graph + 1)
  n_node = np.random.randint(0, 10, n_graph)
  n_edge = np.random.randint(0, 20, n_graph)
  # We cannot have any edges if there are no nodes.
  n_edge[n_node == 0] = 0

  senders = []
  receivers = []
  offset = 0
  for n_node_in_graph, n_edge_in_graph in zip(n_node, n_edge):
    if n_edge_in_graph != 0:
      senders += list(
          np.random.randint(0, n_node_in_graph, n_edge_in_graph) + offset)
      receivers += list(
          np.random.randint(0, n_node_in_graph, n_edge_in_graph) + offset)
    offset += n_node_in_graph
  if include_globals:
    global_features = jnp.asarray(np.random.random(size=(n_graph, 5)))
  else:
    global_features = None
  if include_node_features:
    nodes = jnp.asarray(np.random.random(size=(np.sum(n_node), 4)))
  else:
    nodes = None

  if include_edge_features:
    edges = jnp.asarray(np.random.random(size=(np.sum(n_edge), 3)))
  else:
    edges = None
  return graph.GraphsTuple(
      n_node=jnp.asarray(n_node),
      n_edge=jnp.asarray(n_edge),
      nodes=nodes,
      edges=edges,
      globals=global_features,
      senders=jnp.asarray(senders),
      receivers=jnp.asarray(receivers))


class GraphTest(test_util.JaxTestCase):

  def _make_nest(self, array):
    """Returns a nest given an array."""
    return {'a': array,
            'b': [jnp.ones_like(array), {'c': jnp.zeros_like(array)}]}

  def _get_list_and_batched_graph(self):
    """Returns a list of individual graphs and a batched version.

    This test-case includes the following corner-cases:
      - single node,
      - multiple nodes,
      - no edges,
      - single edge,
      - and multiple edges.
    """
    batched_graph = graph.GraphsTuple(
        n_node=jnp.array([1, 3, 1, 0, 2, 0, 0]),
        n_edge=jnp.array([2, 5, 0, 0, 1, 0, 0]),
        nodes=self._make_nest(jnp.arange(14).reshape(7, 2)),
        edges=self._make_nest(jnp.arange(24).reshape(8, 3)),
        globals=self._make_nest(jnp.arange(14).reshape(7, 2)),
        senders=jnp.array([0, 0, 1, 1, 2, 3, 3, 6]),
        receivers=jnp.array([0, 0, 2, 1, 3, 2, 1, 5]))

    list_graphs = [
        graph.GraphsTuple(
            n_node=jnp.array([1]),
            n_edge=jnp.array([2]),
            nodes=self._make_nest(jnp.array([[0, 1]])),
            edges=self._make_nest(jnp.array([[0, 1, 2], [3, 4, 5]])),
            globals=self._make_nest(jnp.array([[0, 1]])),
            senders=jnp.array([0, 0]),
            receivers=jnp.array([0, 0])),
        graph.GraphsTuple(
            n_node=jnp.array([3]),
            n_edge=jnp.array([5]),
            nodes=self._make_nest(jnp.array([[2, 3], [4, 5], [6, 7]])),
            edges=self._make_nest(
                jnp.array([[6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17],
                           [18, 19, 20]])),
            globals=self._make_nest(jnp.array([[2, 3]])),
            senders=jnp.array([0, 0, 1, 2, 2]),
            receivers=jnp.array([1, 0, 2, 1, 0])),
        graph.GraphsTuple(
            n_node=jnp.array([1]),
            n_edge=jnp.array([0]),
            nodes=self._make_nest(jnp.array([[8, 9]])),
            edges=self._make_nest(jnp.zeros((0, 3))),
            globals=self._make_nest(jnp.array([[4, 5]])),
            senders=jnp.array([]),
            receivers=jnp.array([])),
        graph.GraphsTuple(
            n_node=jnp.array([0]),
            n_edge=jnp.array([0]),
            nodes=self._make_nest(jnp.zeros((0, 2))),
            edges=self._make_nest(jnp.zeros((0, 3))),
            globals=self._make_nest(jnp.array([[6, 7]])),
            senders=jnp.array([]),
            receivers=jnp.array([])),
        graph.GraphsTuple(
            n_node=jnp.array([2]),
            n_edge=jnp.array([1]),
            nodes=self._make_nest(jnp.array([[10, 11], [12, 13]])),
            edges=self._make_nest(jnp.array([[21, 22, 23]])),
            globals=self._make_nest(jnp.array([[8, 9]])),
            senders=jnp.array([1]),
            receivers=jnp.array([0])),
        graph.GraphsTuple(
            n_node=jnp.array([0]),
            n_edge=jnp.array([0]),
            nodes=self._make_nest(jnp.zeros((0, 2))),
            edges=self._make_nest(jnp.zeros((0, 3))),
            globals=self._make_nest(jnp.array([[10, 11]])),
            senders=jnp.array([]),
            receivers=jnp.array([])),
        graph.GraphsTuple(
            n_node=jnp.array([0]),
            n_edge=jnp.array([0]),
            nodes=self._make_nest(jnp.zeros((0, 2))),
            edges=self._make_nest(jnp.zeros((0, 3))),
            globals=self._make_nest(jnp.array([[12, 13]])),
            senders=jnp.array([]),
            receivers=jnp.array([])),
        graph.GraphsTuple(
            n_node=jnp.array([]),
            n_edge=jnp.array([]),
            nodes=self._make_nest(jnp.zeros((0, 2))),
            edges=self._make_nest(jnp.zeros((0, 3))),
            globals=self._make_nest(jnp.zeros((0, 2))),
            senders=jnp.array([]),
            receivers=jnp.array([])),
    ]

    return list_graphs, batched_graph

  def test_batch(self):
    """Tests batching of graph."""
    list_graphs_tuple, batched_graphs_tuple = self._get_list_and_batched_graph()
    graphs_tuple = utils.batch(list_graphs_tuple)
    self.assertAllClose(graphs_tuple, batched_graphs_tuple, check_dtypes=False)

  def test_unbatch(self):
    """Tests unbatching of graph."""
    list_graphs_tuple, batched_graphs_tuple = self._get_list_and_batched_graph()
    graphs_tuples = utils.unbatch(batched_graphs_tuple)
    # The final GraphsTuple does not contain a graph, and so shouldn't be
    # present in the result.
    self.assertAllClose(
        graphs_tuples, list_graphs_tuple[:-1], check_dtypes=False)

  @parameterized.parameters((True, True, False),
                            (True, False, True),
                            (False, True, True))
  def test_batch_unbatch_with_random_graphs(self,
                                            include_globals,
                                            include_nodes,
                                            include_edges):
    """Tests batch(unbatch) is identity with random graphs."""
    np.random.seed(42)
    for _ in range(100):
      g = _get_random_graph(include_globals=include_globals,
                            include_node_features=include_nodes,
                            include_edge_features=include_edges)
      self.assertAllClose(utils.batch(utils.unbatch(g)), g, check_dtypes=True)

    for _ in range(10):
      rg = lambda: _get_random_graph(  # pylint: disable=g-long-lambda
          1, include_nodes, include_edges, include_globals)
      graphs1 = [rg() for _ in range(np.random.randint(1, 10))]
      graphs2 = utils.unbatch(utils.batch(graphs1))
      for g1, g2 in zip(graphs1, graphs2):
        self.assertAllClose(g1, g2, check_dtypes=False)

  def test_pad_with_graphs(self):
    """Tests padding of graph."""
    _, graphs_tuple = self._get_list_and_batched_graph()
    padded_graphs_tuple = utils.pad_with_graphs(graphs_tuple, 10, 12, 9)
    expected_padded_graph = graph.GraphsTuple(
        n_node=jnp.concatenate([graphs_tuple.n_node,
                                jnp.array([3, 0])]),
        n_edge=jnp.concatenate([graphs_tuple.n_edge,
                                jnp.array([4, 0])]),
        nodes=tree.tree_map(
            lambda f: jnp.concatenate([f, jnp.zeros((3, 2), dtype=f.dtype)]),
            graphs_tuple.nodes),
        edges=tree.tree_map(
            lambda f: jnp.concatenate([f, jnp.zeros((4, 3), dtype=f.dtype)]),
            graphs_tuple.edges),
        globals=tree.tree_map(
            lambda f: jnp.concatenate([f, jnp.zeros((2, 2), dtype=f.dtype)]),
            graphs_tuple.globals),
        senders=jnp.concatenate([graphs_tuple.senders,
                                 jnp.array([7, 7, 7, 7])]),
        receivers=jnp.concatenate(
            [graphs_tuple.receivers,
             jnp.array([7, 7, 7, 7])]),
    )
    self.assertAllClose(
        padded_graphs_tuple, expected_padded_graph, check_dtypes=True)

  def test_unpad(self):
    """Tests unpadding of graph."""
    _, graphs_tuple = self._get_list_and_batched_graph()
    unpadded_graphs_tuple = utils.unpad_with_graphs(graphs_tuple)
    expected_unpadded_graph = graph.GraphsTuple(
        n_node=jnp.array([1, 3, 1, 0]),
        n_edge=jnp.array([2, 5, 0, 0]),
        nodes=self._make_nest(jnp.arange(10).reshape(5, 2)),
        edges=self._make_nest(jnp.arange(21).reshape(7, 3)),
        globals=self._make_nest(jnp.arange(8).reshape(4, 2)),
        senders=jnp.array([0, 0, 1, 1, 2, 3, 3]),
        receivers=jnp.array([0, 0, 2, 1, 3, 2, 1]))
    self.assertAllClose(
        unpadded_graphs_tuple, expected_unpadded_graph, check_dtypes=True)

  @parameterized.parameters((True, True, False),
                            (True, False, True),
                            (False, True, True))
  def test_pad_unpad_with_random_graphs(self,
                                        include_globals,
                                        include_nodes,
                                        include_edges):
    """Tests unpad(pad) is identity with random graphs."""
    np.random.seed(42)
    for _ in range(100):
      g = _get_random_graph(include_globals=include_globals,
                            include_node_features=include_nodes,
                            include_edge_features=include_edges)
      self.assertAllClose(
          utils.unpad_with_graphs(utils.pad_with_graphs(g, 101, 200, 11)),
          g, check_dtypes=True)

  def test_get_number_of_padding_with_graphs_graphs(self):
    """Tests the number of padding graphs calculation."""
    _, graphs_tuple = self._get_list_and_batched_graph()
    expected = 3
    with self.subTest('nojit'):
      self.assertAllClose(
          utils.get_number_of_padding_with_graphs_graphs(graphs_tuple),
          expected,
          check_dtypes=True)
    with self.subTest('jit'):
      self.assertAllClose(
          jax.jit(utils.get_number_of_padding_with_graphs_graphs)(graphs_tuple),
          expected,
          check_dtypes=True)

  def test_get_number_of_padding_with_graphs_nodes(self):
    """Tests the number of padding nodes calculation."""
    _, graphs_tuple = self._get_list_and_batched_graph()
    expected = 2
    with self.subTest('nojit'):
      self.assertAllClose(
          utils.get_number_of_padding_with_graphs_nodes(graphs_tuple),
          expected,
          check_dtypes=True)
    with self.subTest('jit'):
      self.assertAllClose(
          jax.jit(utils.get_number_of_padding_with_graphs_nodes)(graphs_tuple),
          expected,
          check_dtypes=True)

  def test_get_number_of_padding_with_graphs_edges(self):
    """Tests the number of padding edges calculation."""
    _, graphs_tuple = self._get_list_and_batched_graph()
    expected = 1
    with self.subTest('nojit'):
      self.assertAllClose(
          utils.get_number_of_padding_with_graphs_edges(graphs_tuple),
          expected,
          check_dtypes=True)
    with self.subTest('jit'):
      self.assertAllClose(
          jax.jit(utils.get_number_of_padding_with_graphs_edges)(graphs_tuple),
          expected,
          check_dtypes=True)

  def test_get_node_padding_mask(self):
    """Tests construction of node padding mask."""
    _, graphs_tuple = self._get_list_and_batched_graph()
    expected_mask = jnp.array([1, 1, 1, 1, 1, 0, 0]).astype(bool)
    with self.subTest('nojit'):
      mask = utils.get_node_padding_mask(graphs_tuple)
      self.assertArraysEqual(mask, expected_mask)
    with self.subTest('jit'):
      mask = jax.jit(utils.get_node_padding_mask)(graphs_tuple)
      self.assertArraysEqual(mask, expected_mask)

  def test_get_edge_padding_mask(self):
    """Tests construction of edge padding mask."""
    _, graphs_tuple = self._get_list_and_batched_graph()
    expected_mask = jnp.array([1, 1, 1, 1, 1, 1, 1, 0]).astype(bool)
    with self.subTest('nojit'):
      mask = utils.get_edge_padding_mask(graphs_tuple)
      self.assertArraysEqual(mask, expected_mask)
    with self.subTest('jit'):
      mask = jax.jit(utils.get_edge_padding_mask)(graphs_tuple)
      self.assertArraysEqual(mask, expected_mask)

  def test_get_graph_padding_mask(self):
    """Tests construction of graph padding mask."""
    _, graphs_tuple = self._get_list_and_batched_graph()
    expected_mask = jnp.array([1, 1, 1, 1, 0, 0, 0]).astype(bool)
    with self.subTest('nojit'):
      mask = utils.get_graph_padding_mask(graphs_tuple)
      self.assertArraysEqual(mask, expected_mask)
    with self.subTest('jit'):
      mask = jax.jit(utils.get_graph_padding_mask)(graphs_tuple)
      self.assertArraysEqual(mask, expected_mask)

  def test_segment_sum(self):
    result = utils.segment_sum(
      jnp.arange(9), jnp.array([0, 1, 2, 0, 4, 0, 1, 1, 0]), 6)
    self.assertAllClose(result, jnp.array([16, 14, 2, 0, 4, 0]),
                        check_dtypes=False)

  def test_segment_mean(self):
    result = utils.segment_mean(
      jnp.arange(9), jnp.array([0, 1, 2, 0, 4, 0, 1, 1, 0]), 6)
    self.assertAllClose(result, jnp.array([4, 14 / 3.0, 2, 0, 4, 0]),
                        check_dtypes=False)

  def test_segment_variance(self):
    result = utils.segment_variance(
      jnp.arange(8), jnp.array([0, 0, 0, 1, 1, 2, 2, 2]), 3)
    self.assertAllClose(result, jnp.stack([jnp.var(jnp.arange(3)),
                                           jnp.var(jnp.arange(3, 5)),
                                           jnp.var(jnp.arange(5, 8))]))

  def test_segment_normalize(self):
    result = utils.segment_normalize(
      jnp.arange(8), jnp.array([0, 0, 0, 1, 1, 2, 2, 2]), 3)
    self.assertAllClose(result,
                        jnp.concatenate([jax.nn.normalize(jnp.arange(3)),
                                         jax.nn.normalize(jnp.arange(3, 5)),
                                         jax.nn.normalize(jnp.arange(5, 8))]))

  @parameterized.parameters((False, False),
                            (True, False),
                            (True, True),
                            (False, True))
  def test_segment_max(self, indices_are_sorted, unique_indices):
    neg_inf = jnp.iinfo(jnp.int32).min
    if unique_indices:
      data = jnp.arange(6)
      if indices_are_sorted:
        segment_ids = jnp.array([0, 1, 2, 3, 4, 5])
        expected_out = jnp.array([0, 1, 2, 3, 4, 5])
        num_segments = 6
      else:
        segment_ids = jnp.array([1, 0, 2, 4, 3, -5])
        expected_out = jnp.array([5, 0, 2, 4, 3])
        num_segments = 5
    else:
      data = jnp.arange(9)
      if indices_are_sorted:
        segment_ids = jnp.array([0, 0, 0, 1, 1, 1, 2, 3, 4])
        expected_out = jnp.array([2, 5, 6, 7, 8, neg_inf])
      else:
        segment_ids = jnp.array([0, 1, 2, 0, 4, 0, 1, 1, -6])
        expected_out = jnp.array([8, 7, 2, neg_inf, 4, neg_inf])
      num_segments = 6

    with self.subTest('nojit'):
      result = utils.segment_max(data, segment_ids, num_segments,
                                 indices_are_sorted, unique_indices)
      self.assertAllClose(result, expected_out, check_dtypes=True)
      result = utils.segment_max(data, segment_ids,
                                 indices_are_sorted=indices_are_sorted,
                                 unique_indices=unique_indices)
      num_unique_segments = jnp.maximum(jnp.max(segment_ids) + 1,
                                        jnp.max(-segment_ids))
      self.assertAllClose(result, expected_out[:num_unique_segments],
                          check_dtypes=True)
    with self.subTest('jit'):
      result = jax.jit(utils.segment_max, static_argnums=(2, 3, 4))(
          data, segment_ids, num_segments, indices_are_sorted, unique_indices)
      self.assertAllClose(result, expected_out, check_dtypes=True)

  @parameterized.parameters((False, False),
                            (True, False),
                            (True, True),
                            (False, True))
  def test_segment_max_negatives(self, indices_are_sorted, unique_indices):
    neg_inf = jnp.iinfo(jnp.int32).min
    if unique_indices:
      data = -1 - jnp.arange(6)  # [-1, -2, -3, -4, -5, -6]
      if indices_are_sorted:
        segment_ids = jnp.array([0, 1, 2, 3, 4, 5])
        expected_out = jnp.array([-1, -2, -3, -4, -5, -6])
        num_segments = 6
      else:
        segment_ids = jnp.array([1, 0, 2, 4, 3, -5])
        expected_out = jnp.array([-2, -1, -3, -5, -4])
        num_segments = 5
    else:
      data = -1 - jnp.arange(9)  # [-1, -2, -3, -4, -5, -6, -7, -8, -9]
      if indices_are_sorted:
        segment_ids = jnp.array([0, 0, 0, 1, 1, 1, 2, 3, 4])
        expected_out = jnp.array([-1, -4, -7, -8, -9, neg_inf])
      else:
        segment_ids = jnp.array([0, 1, 2, 0, 4, 0, 1, 1, -6])
        expected_out = jnp.array([-1, -2, -3, neg_inf, -5, neg_inf])
      num_segments = 6

    with self.subTest('nojit'):
      result = utils.segment_max(data, segment_ids, num_segments,
                                 indices_are_sorted, unique_indices)
      self.assertAllClose(result, expected_out, check_dtypes=True)
      result = utils.segment_max(data, segment_ids,
                                 indices_are_sorted=indices_are_sorted,
                                 unique_indices=unique_indices)
      num_unique_segments = jnp.maximum(jnp.max(segment_ids) + 1,
                                        jnp.max(-segment_ids))
      self.assertAllClose(result, expected_out[:num_unique_segments],
                          check_dtypes=True)
    with self.subTest('jit'):
      result = jax.jit(utils.segment_max, static_argnums=(2, 3, 4))(
          data, segment_ids, num_segments, indices_are_sorted, unique_indices)
      self.assertAllClose(result, expected_out, check_dtypes=True)

  def test_segment_softmax(self):
    data = jnp.arange(9)
    segment_ids = jnp.array([0, 1, 2, 0, 4, 0, 1, 1, 0])
    num_segments = 6
    expected_out = np.array([3.1741429e-04, 1.8088353e-03, 1.0000000e+00,
                             6.3754367e-03, 1.0000000e+00, 4.7108460e-02,
                             2.6845494e-01, 7.2973621e-01, 9.4619870e-01])
    with self.subTest('nojit'):
      result = utils.segment_softmax(data, segment_ids, num_segments)
      self.assertAllClose(result, expected_out, check_dtypes=True)
      result = utils.segment_softmax(data, segment_ids)
      self.assertAllClose(result, expected_out, check_dtypes=True)
    with self.subTest('jit'):
      result = jax.jit(
          utils.segment_softmax, static_argnums=2)(data, segment_ids,
                                                   num_segments)
      self.assertAllClose(result, expected_out, check_dtypes=True)

  def test_partition_softmax(self):
    data = jnp.arange(9)
    partitions = jnp.array([3, 2, 4])
    sum_partitions = 9
    expected_out = np.array([0.090031, 0.244728, 0.665241, 0.268941, 0.731059,
                             0.032059, 0.087144, 0.236883, 0.643914])
    with self.subTest('nojit'):
      result = utils.partition_softmax(data, partitions, sum_partitions)
      self.assertAllClose(result, expected_out, check_dtypes=True)
      result = utils.partition_softmax(data, partitions)
      self.assertAllClose(result, expected_out, check_dtypes=True)
    with self.subTest('jit'):
      result = jax.jit(utils.partition_softmax, static_argnums=2)(
          data, partitions, sum_partitions)
      self.assertAllClose(result, expected_out, check_dtypes=True)


class ConcatenatedArgsWrapperTest(test_util.JaxTestCase):

  @parameterized.parameters(
      ([], {'a': np.array([10, 2])}, -1),
      ([np.array([10, 5])], {'a': np.array([10, 2])}, -1),
      ([np.array([10, 5]), np.array([10, 3])], {'a': np.array([10, 2])}, -1),
      ([np.array([10, 5]), np.array([10, 3])], {}, -1),
      ([{'a': np.array([10, 2]), 'b': np.array([10, 4])}],
       {'c': np.array([10, 3])}, 1),
      ([{'a': np.array([2, 10]), 'b': np.array([4, 10])}],
       {'c': np.array([3, 10])}, 0))
  def test_single_arg(self, args_shapes, kwargs_shapes, axis):
    args = tree.tree_map(lambda x: np.random.randn(*x), args_shapes)
    kwargs = {k: np.random.randn(*shape) for k, shape in kwargs_shapes.items()}

    @utils.concatenated_args(axis=axis)
    def update_fn(feat):
      return feat

    out = update_fn(*args, **kwargs)
    expected_out = jnp.concatenate(
        list(tree.tree_flatten(args)[0]) + list(tree.tree_flatten(kwargs)[0]),
        axis=axis)
    self.assertArraysAllClose(out, expected_out, check_dtypes=True)


if __name__ == '__main__':
  absltest.main()
