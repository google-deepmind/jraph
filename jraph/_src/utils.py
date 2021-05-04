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
"""Utilities for working with `GraphsTuple`s."""

import functools
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Union

import jax
from jax import lax
import jax.numpy as jnp
import jax.tree_util as tree
from jraph._src import graph as gn_graph
import numpy as np


# As of 04/2020 pytype doesn't support recursive types.
# pytype: disable=not-supported-yet
ArrayTree = Union[jnp.ndarray, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]


# Defines segment_sum for the convenience of the user.
segment_sum = jax.ops.segment_sum


# Defines a sorted segment_sum for the convenience of the user. A sorted segment
# sum provide additional optimizations over unsorted segment_sum on some
# backends. Improvements can be noticed especially for large sums on TPUs.
sorted_segment_sum = functools.partial(segment_sum, indices_are_sorted=True)


def segment_mean(data, segment_ids, num_segments):
  """Returns mean for each segment.

  Args:
    data: the values which are averaged segment-wise.
    segment_ids: indices for the segments.
    num_segments: total number of segments.
  """
  nominator = jax.ops.segment_sum(data, segment_ids, num_segments)
  denominator = jax.ops.segment_sum(
      jnp.ones_like(data), segment_ids, num_segments)
  return jnp.nan_to_num(nominator / denominator)


def segment_variance(data, segment_ids, num_segments):
  """Returns the variance for each segment.

  Args:
    data: values whose variance will be calculated segment-wise.
    segment_ids: indices for segments
    num_segments: total number of segments.

  Returns:
    num_segments size array containing the variance of each segment.
  """
  means = segment_mean(data, segment_ids, num_segments)[segment_ids]
  counts = jax.ops.segment_sum(jnp.ones_like(data), segment_ids, num_segments)
  variances = jax.ops.segment_sum(jnp.power(data - means, 2),
                                  segment_ids, num_segments) / counts
  return jnp.nan_to_num(variances)


def segment_normalize(data, segment_ids, num_segments, eps=1e-5):
  """Normalizes data within each segment.

  Args:
    data: values whose z-score normalized values will be calculated.
      segment-wise.
    segment_ids: indices for segments.
    num_segments: total number of segments.
    eps: epsilon for numerical stability.

  Returns:
    array containing data normalized segment-wise.
  """
  means = segment_mean(data, segment_ids, num_segments)[segment_ids]
  variances = segment_variance(data, segment_ids, num_segments)[segment_ids]
  normalized = (data - means) * lax.rsqrt(variances + eps)
  return jnp.nan_to_num(normalized)


def batch(graphs: Sequence[gn_graph.GraphsTuple]) -> gn_graph.GraphsTuple:
  """Returns a batched graph given a list of graphs.

  This method will concatenate the ``nodes``, ``edges`` and ``globals``,
  ``n_node`` and ``n_edge`` of a sequence of ``GraphsTuple`` along axis 0. For
  ``senders`` and ``receivers``, offsets are computed so that connectivity
  remains valid for the new node indices.

  For example::

    key = jax.random.PRNGKey(0)
    graph_1 = GraphsTuple(nodes=jax.random.normal(key, (3, 64)),
                      edges=jax.random.normal(key, (5, 64)),
                      senders=jnp.array([0,0,1,1,2]),
                      receivers=[1,2,0,2,1],
                      n_node=jnp.array([3]),
                      n_edge=jnp.array([5]),
                      globals=jax.random.normal(key, (1, 64)))
    graph_2 = GraphsTuple(nodes=jax.random.normal(key, (5, 64)),
                      edges=jax.random.normal(key, (10, 64)),
                      senders=jnp.array([0,0,1,1,2,2,3,3,4,4]),
                      receivers=jnp.array([1,2,0,2,1,0,2,1,3,2]),
                      n_node=jnp.array([5]),
                      n_edge=jnp.array([10]),
                      globals=jax.random.normal(key, (1, 64)))
    batch = graph.batch([graph_1, graph_2])

    batch.nodes.shape
    >> (8, 64)
    batch.edges.shape
    >> (15, 64)
    # Offsets computed on senders and receivers
    batch.senders
    >> DeviceArray([0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7], dtype=int32)
    batch.receivers
    >> DeviceArray([1, 2, 0, 2, 1, 4, 5, 3, 5, 4, 3, 5, 4, 6, 5], dtype=int32)
    batch.n_node
    >> DeviceArray([3, 5], dtype=int32)
    batch.n_edge
    >> DeviceArray([5, 10], dtype=int32)

  If a ``GraphsTuple`` does not contain any graphs, it will be dropped from the
  batch.

  This method is not compilable because it is data dependent.

  This functionality was implementation as  ``utils_tf.concat`` in the
  Tensorflow version of graph_nets.

  Args:
    graphs: sequence of ``GraphsTuple``s which will be batched into a single
      graph.
  """
  return _batch(graphs, np_=jnp)


def _batch(graphs, np_):
  """Returns batched graph given a list of graphs and a numpy-like module."""
  # Calculates offsets for sender and receiver arrays, caused by concatenating
  # the nodes arrays.
  offsets = np_.cumsum(
      np_.array([0] + [np_.sum(g.n_node) for g in graphs[:-1]]))

  def _map_concat(nests):
    concat = lambda *args: np_.concatenate(args)
    return tree.tree_multimap(concat, *nests)

  return gn_graph.GraphsTuple(
      n_node=np_.concatenate([g.n_node for g in graphs]),
      n_edge=np_.concatenate([g.n_edge for g in graphs]),
      nodes=_map_concat([g.nodes for g in graphs]),
      edges=_map_concat([g.edges for g in graphs]),
      globals=_map_concat([g.globals for g in graphs]),
      senders=np_.concatenate([g.senders + o for g, o in zip(graphs, offsets)]),
      receivers=np_.concatenate(
          [g.receivers + o for g, o in zip(graphs, offsets)]))


def unbatch(graph: gn_graph.GraphsTuple) -> List[gn_graph.GraphsTuple]:
  """Returns a list of graphs given a batched graph.

  This function does not support jax.jit, because the shape of the output
  is data-dependent!

  Args:
    graph: the batched graph, which will be unbatched into a list of graphs.
  """
  return _unbatch(graph, np_=jnp)


def _unbatch(
    graph: gn_graph.GraphsTuple, np_) -> List[gn_graph.GraphsTuple]:
  """Returns a list of graphs given a batched graph."""

  def _map_split(nest, indices_or_sections):
    """Splits leaf nodes of nests and returns a list of nests."""
    if isinstance(indices_or_sections, int):
      n_lists = indices_or_sections
    else:
      n_lists = len(indices_or_sections) + 1
    concat = lambda field: np_.split(field, indices_or_sections)
    nest_of_lists = tree.tree_map(concat, nest)
    # pylint: disable=cell-var-from-loop
    list_of_nests = [tree.tree_multimap(lambda _, x: x[i], nest, nest_of_lists)
                     for i in range(n_lists)]
    return list_of_nests

  all_n_node = graph.n_node[:, None]
  all_n_edge = graph.n_edge[:, None]
  node_offsets = np_.cumsum(graph.n_node[:-1])
  all_nodes = _map_split(graph.nodes, node_offsets)
  edge_offsets = np_.cumsum(graph.n_edge[:-1])
  all_edges = _map_split(graph.edges, edge_offsets)
  all_globals = _map_split(graph.globals, len(graph.n_node))
  all_senders = np_.split(graph.senders, edge_offsets)
  all_receivers = np_.split(graph.receivers, edge_offsets)

  # Corrects offset in the sender and receiver arrays, caused by splitting the
  # nodes array.
  n_graphs = graph.n_node.shape[0]
  for graph_index in np_.arange(n_graphs)[1:]:
    all_senders[graph_index] -= node_offsets[graph_index - 1]
    all_receivers[graph_index] -= node_offsets[graph_index - 1]

  return [gn_graph.GraphsTuple._make(elements)
          for elements in zip(all_nodes, all_edges, all_receivers, all_senders,
                              all_globals, all_n_node, all_n_edge)]


def pad_with_graphs(graph: gn_graph.GraphsTuple,
                    n_node: int,
                    n_edge: int,
                    n_graph: int = 2) -> gn_graph.GraphsTuple:
  """Pads a ``GraphsTuple`` to size by adding computation preserving graphs.

  The ``GraphsTuple`` is padded by first adding a dummy graph which contains the
  padding nodes and edges, and then empty graphs without nodes or edges.

  The empty graphs and the dummy graph do not interfer with the graphnet
  calculations on the original graph, and so are computation preserving.

  The padding graph requires at least one node and one graph.

  This function does not support jax.jit, because the shape of the output
  is data-dependent.

  Args:
    graph: ``GraphsTuple`` padded with dummy graph and empty graphs.
    n_node: the number of nodes in the padded ``GraphsTuple``.
    n_edge: the number of edges in the padded ``GraphsTuple``.
    n_graph: the number of graphs in the padded ``GraphsTuple``. Default is 2,
      which is the lowest possible value, because we always have at least one
      graph in the original ``GraphsTuple`` and we need one dummy graph for the
      padding.

  Raises:
    ValueError: if the passed ``n_graph`` is smaller than 2.
    RuntimeError: if the given ``GraphsTuple`` is too large for the given
      padding.

  Returns:
    A padded ``GraphsTuple``.
  """
  if n_graph < 2:
    raise ValueError(
        f'n_graph is {n_graph}, which is smaller than minimum value of 2.')
  graph = jax.device_get(graph)
  pad_n_node = int(n_node - np.sum(graph.n_node))
  pad_n_edge = int(n_edge - np.sum(graph.n_edge))
  pad_n_graph = int(n_graph - graph.n_node.shape[0])
  if pad_n_node <= 0 or pad_n_edge < 0 or pad_n_graph <= 0:
    raise RuntimeError(
        'Given graph is too large for the given padding. difference: '
        f'n_node {pad_n_node}, n_edge {pad_n_edge}, n_graph {pad_n_graph}')

  pad_n_empty_graph = pad_n_graph - 1

  tree_nodes_pad = (
      lambda leaf: np.zeros((pad_n_node,) + leaf.shape[1:], dtype=leaf.dtype))
  tree_edges_pad = (
      lambda leaf: np.zeros((pad_n_edge,) + leaf.shape[1:], dtype=leaf.dtype))
  tree_globs_pad = (
      lambda leaf: np.zeros((pad_n_graph,) + leaf.shape[1:], dtype=leaf.dtype))

  padding_graph = gn_graph.GraphsTuple(
      n_node=np.concatenate([np.array([pad_n_node]),
                             np.zeros(pad_n_empty_graph, dtype=np.int32)]),
      n_edge=np.concatenate([np.array([pad_n_edge]),
                             np.zeros(pad_n_empty_graph, dtype=np.int32)]),
      nodes=tree.tree_map(tree_nodes_pad, graph.nodes),
      edges=tree.tree_map(tree_edges_pad, graph.edges),
      globals=tree.tree_map(tree_globs_pad, graph.globals),
      senders=np.zeros(pad_n_edge, dtype=np.int32),
      receivers=np.zeros(pad_n_edge, dtype=np.int32),
  )
  return _batch([graph, padding_graph], np_=np)


def get_number_of_padding_with_graphs_graphs(
    padded_graph: gn_graph.GraphsTuple) -> int:
  """Returns number of padding graphs in padded_graph.

  Warning: This method only gives results for graphs that have been padded with
  ``pad_with_graphs``.

  Args:
    padded_graph: a ``GraphsTuple`` that has been padded with
      ``pad_with_graphs``.

  Returns:
    The number of padding graphs.
  """
  # The first_padding graph always has at least one padding node, and
  # all padding graphs that follow have 0 nodes. We can count how many
  # trailing graphs have 0 nodes, and add one.
  n_trailing_empty_padding_graphs = jnp.argmin(padded_graph.n_node[::-1] == 0)
  return n_trailing_empty_padding_graphs + 1


def get_number_of_padding_with_graphs_nodes(
    padded_graph: gn_graph.GraphsTuple) -> int:
  """Returns number of padding nodes in given padded_graph.

  Warning: This method only gives results for graphs that have been padded with
  ``pad_with_graphs``.

  Args:
    padded_graph: a ``GraphsTuple`` that has been padded with
      ``pad_with_graphs``.

  Returns:
    The number of padding nodes.
  """
  return padded_graph.n_node[
      -get_number_of_padding_with_graphs_graphs(padded_graph)]


def get_number_of_padding_with_graphs_edges(
    padded_graph: gn_graph.GraphsTuple) -> int:
  """Returns number of padding edges in given padded_graph.

  Warning: This method only gives results for graphs that have been padded with
  ``pad_with_graphs``.

  Args:
    padded_graph: a ``GraphsTuple`` that has been padded with
      ``pad_with_graphs``.

  Returns:
    The number of padding edges.
  """
  return padded_graph.n_edge[
      -get_number_of_padding_with_graphs_graphs(padded_graph)]


def unpad_with_graphs(
    padded_graph: gn_graph.GraphsTuple) -> gn_graph.GraphsTuple:
  """Unpads the given graph by removing the dummy graph and empty graphs.

  This function assumes that the given graph was padded with the
  ``pad_with_graphs`` function.

  This function does not support jax.jit, because the shape of the output
  is data-dependent!

  Args:
    padded_graph: ``GraphsTuple`` padded with a dummy graph
      and empty graphs.

  Returns:
    The unpadded graph.
  """
  n_padding_graph = get_number_of_padding_with_graphs_graphs(padded_graph)
  n_padding_node = get_number_of_padding_with_graphs_nodes(padded_graph)
  n_padding_edge = get_number_of_padding_with_graphs_edges(padded_graph)

  unpadded_graph = gn_graph.GraphsTuple(
      n_node=padded_graph.n_node[:-n_padding_graph],
      n_edge=padded_graph.n_edge[:-n_padding_graph],
      nodes=tree.tree_map(lambda x: x[:-n_padding_node], padded_graph.nodes),
      edges=tree.tree_map(lambda x: x[:-n_padding_edge], padded_graph.edges),
      globals=tree.tree_map(lambda x: x[:-n_padding_graph],
                            padded_graph.globals),
      senders=padded_graph.senders[:-n_padding_edge],
      receivers=padded_graph.receivers[:-n_padding_edge],
  )
  return unpadded_graph


def get_node_padding_mask(padded_graph: gn_graph.GraphsTuple) -> ArrayTree:
  """Returns a mask for the nodes of a padded graph.

  Args:
    padded_graph: ``GraphsTuple`` padded using ``pad_with_graphs``. This
        graph must contain at least one array of node features so the total
        static number of nodes can be inferred statically from the shape, and
        the method can be jitted.

  Returns:
    Boolean array of shape [total_num_nodes] containing True for real nodes,
    and False for padding nodes.
  """
  n_padding_node = get_number_of_padding_with_graphs_nodes(padded_graph)
  flat_node_features = tree.tree_leaves(padded_graph.nodes)

  if not flat_node_features:
    raise ValueError(
        '`padded_graph` must have at least one array of node features')
  total_num_nodes = flat_node_features[0].shape[0]
  return _get_mask(padding_length=n_padding_node, full_length=total_num_nodes)


def get_edge_padding_mask(padded_graph: gn_graph.GraphsTuple) -> ArrayTree:
  """Returns a mask for the edges of a padded graph.

  Args:
    padded_graph: ``GraphsTuple`` padded using ``pad_with_graphs``.

  Returns:
    Boolean array of shape [total_num_edges] containing True for real edges,
    and False for padding edges.
  """
  n_padding_edge = get_number_of_padding_with_graphs_edges(padded_graph)
  total_num_edges = padded_graph.senders.shape[0]
  return _get_mask(padding_length=n_padding_edge, full_length=total_num_edges)


def get_graph_padding_mask(padded_graph: gn_graph.GraphsTuple) -> ArrayTree:
  """Returns a mask for the graphs of a padded graph.

  Args:
    padded_graph: ``GraphsTuple`` padded using ``pad_with_graphs``.

  Returns:
    Boolean array of shape [total_num_graphs] containing True for real graphs,
    and False for padding graphs.
  """
  n_padding_graph = get_number_of_padding_with_graphs_graphs(padded_graph)
  total_num_graphs = padded_graph.n_node.shape[0]
  return _get_mask(padding_length=n_padding_graph, full_length=total_num_graphs)


def _get_mask(padding_length, full_length):
  valid_length = full_length - padding_length
  return jnp.arange(full_length, dtype=jnp.int32) < valid_length


def concatenated_args(
    update: Optional[Callable[..., ArrayTree]] = None, *,
    axis: int = -1) -> Union[Callable[..., ArrayTree],
                             Callable[[Callable[..., ArrayTree]], ArrayTree]]:
  """Decorator that concatenates arguments before being passed to an update_fn.

  By default node, edge and global features are passed separately to update
  functions. However, it is common practice to concatenate these features before
  passing them to a neural network. This wrapper concatenates the arguments
  for you.

  For example::

    # Without the wrapper
    def update_node_fn(nodes, receivers, globals):
      return net(jnp.concatenate([nodes, receivers, globals], axis=1))

    # With the wrapper
    @concatenated_args
    def update_node_fn(features):
      return net(features)

  Args:
    update: an update function that takes ``jnp.ndarray``.
    axis: the axis upon which to concatenate.

  Returns:
    A wrapped function with the arguments concatenated.
  """
  def _decorate(f):
    @functools.wraps(update)
    def wrapper(*args, **kwargs):
      combined_args = tree.tree_flatten(args)[0] + tree.tree_flatten(kwargs)[0]
      concat_args = jnp.concatenate(combined_args, axis=axis)
      return f(concat_args)
    return wrapper

  # If the update function is passed, then decorate the update function.
  if update:
    return _decorate(update)

  # Otherwise, return the decorator.
  return _decorate


def dtype_max_value(dtype):
  if dtype.kind == 'f':
    return jnp.inf
  elif dtype.kind == 'i':
    return jnp.iinfo(dtype).max
  elif dtype.kind == 'b':
    return True
  else:
    raise ValueError(f'Invalid data type {dtype.kind!r}.')


def dtype_min_value(dtype):
  if dtype.kind == 'f':
    return -jnp.inf
  elif dtype.kind == 'i':
    return jnp.iinfo(dtype).min
  elif dtype.kind == 'b':
    return False
  else:
    raise ValueError(f'Invalid data type {dtype.kind!r}.')


def segment_max(data, segment_ids, num_segments=None,
                indices_are_sorted=False, unique_indices=False):
  """Computes the max within segments of an array.

  Similar to TensorFlow's segment_max:
  https://www.tensorflow.org/api_docs/python/tf/math/segment_max

  Args:
    data: an array with the values to be maxed over.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis) to be maxed over. Values can be repeated
      and need not be sorted. Values outside of the range [0, num_segments) are
      wrapped into that range by applying jnp.mod.
    num_segments: optional, an int with positive value indicating the number of
      segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
      jnp.max(-segment_ids))`` but since `num_segments` determines the size of
      the output, a static value must be provided to use ``segment_max`` in a
      ``jit``-compiled function.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted
    unique_indices: whether ``segment_ids`` is known to be free of duplicates

  Returns:
    An array with shape ``(num_segments,) + data.shape[1:]`` representing
    the segment maxs.
  """
  if num_segments is None:
    num_segments = jnp.maximum(jnp.max(segment_ids) + 1, jnp.max(-segment_ids))
  num_segments = int(num_segments)

  min_value = dtype_min_value(data.dtype)
  out = jnp.full((num_segments,) + data.shape[1:], min_value, dtype=data.dtype)
  segment_ids = jnp.mod(segment_ids, num_segments)
  return jax.ops.index_max(
      out, segment_ids, data, indices_are_sorted, unique_indices)


def segment_min(data, segment_ids, num_segments=None,
                indices_are_sorted=False, unique_indices=False):
  """Computes the min within segments of an array.

  Similar to TensorFlow's segment_min:
  https://www.tensorflow.org/api_docs/python/tf/math/segment_min

  Args:
    data: an array with the values to be maxed over.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis) to be min'd over. Values can be repeated
      and need not be sorted. Values outside of the range [0, num_segments) are
      wrapped into that range by applying jnp.mod.
    num_segments: optional, an int with positive value indicating the number of
      segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
      jnp.max(-segment_ids))`` but since `num_segments` determines the size of
      the output, a static value must be provided to use ``segment_max`` in a
      ``jit``-compiled function.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted
    unique_indices: whether ``segment_ids`` is known to be free of duplicates

  Returns:
    An array with shape ``(num_segments,) + data.shape[1:]`` representing
    the segment mins.
  """
  if num_segments is None:
    num_segments = jnp.maximum(jnp.max(segment_ids) + 1, jnp.max(-segment_ids))
  num_segments = int(num_segments)

  max_value = dtype_max_value(data.dtype)
  out = jnp.full((num_segments,) + data.shape[1:], max_value, dtype=data.dtype)
  segment_ids = jnp.mod(segment_ids, num_segments)
  return jax.ops.index_min(
      out, segment_ids, data, indices_are_sorted, unique_indices)


def segment_softmax(logits: jnp.ndarray, segment_ids: jnp.ndarray,
                    num_segments: Optional[int] = None,
                    indices_are_sorted: bool = False,
                    unique_indices: bool = False) -> ArrayTree:
  """Computes a segment-wise softmax.

  For a given tree of logits that can be divded into segments, computes a
  softmax over the segments.

    logits = jnp.ndarray([1.0, 2.0, 3.0, 1.0, 2.0])
    segment_ids = jnp.ndarray([0, 0, 0, 1, 1])
    segment_softmax(logits, segments)
    >> DeviceArray([0.09003057, 0.24472848, 0.66524094, 0.26894142, 0.7310586],
    >> dtype=float32)

  Args:
    logits: an array of logits to be segment softmaxed.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis) to be maxed over. Values can be repeated
      and need not be sorted. Values outside of the range [0, num_segments) are
      wrapped into that range by applying jnp.mod.
    num_segments: optional, an int with positive value indicating the number of
      segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
      jnp.max(-segment_ids))`` but since ``num_segments`` determines the size of
      the output, a static value must be provided to use ``segment_sum`` in a
      ``jit``-compiled function.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted
    unique_indices: whether ``segment_ids`` is known to be free of duplicates

  Returns:
    The segment softmax-ed ``logits``.
  """
  # First, subtract the segment max for numerical stability
  maxs = segment_max(logits, segment_ids, num_segments, indices_are_sorted,
                     unique_indices)
  logits = logits - maxs[segment_ids]
  # Then take the exp
  logits = jnp.exp(logits)
  # Then calculate the normalizers
  normalizers = jax.ops.segment_sum(logits, segment_ids, num_segments,
                                    indices_are_sorted, unique_indices)
  softmax = logits / normalizers[segment_ids]
  return jnp.nan_to_num(softmax)


def partition_softmax(logits: ArrayTree, partitions: jnp.ndarray,
                      sum_partitions: Optional[int] = None):
  """Compute a softmax within partitions of an array.

    For example::
      logits = jnp.ndarray([1.0, 2.0, 3.0, 1.0, 2.0])
      partitions = jnp.ndarray([3, 2])
      partition_softmax(node_logits, n_node)
      >> DeviceArray(
      >> [0.09003057, 0.24472848, 0.66524094, 0.26894142, 0.7310586],
      >> dtype=float32)

  Args:
    logits: the logits for the softmax.
    partitions: the number of nodes per graph. It is a vector of integers with
      shape ``[n_graphs]``, such that ``graph.n_node[i]`` is the number of nodes
      in the i-th graph.
    sum_partitions: the sum of n_node. If not passed, the result of this method
      is data dependent and so not ``jit``-able.

  Returns:
    The softmax over nodes by graph.
  """
  n_partitions = len(partitions)
  segment_ids = jnp.repeat(jnp.arange(n_partitions), partitions, axis=0,
                           total_repeat_length=sum_partitions)
  return segment_softmax(logits, segment_ids, n_partitions,
                         indices_are_sorted=True)


def get_fully_connected_graph(n_node_per_graph: int,
                              n_graph: int,
                              node_features: Optional[ArrayTree] = None,
                              global_features: Optional[ArrayTree] = None,
                              add_self_edges: bool = True):
  """Gets a fully connected graph given n_node_per_graph and n_graph.

  This method is jittable.

  Args:
    n_node_per_graph: The number of nodes in each graph.
    n_graph: The number of graphs in the `jraph.GraphsTuple`.
    node_features: Optional node features.
    global_features: Optional global features.
    add_self_edges: Whether to add self edges to the graph.

  Returns:
    `jraph.GraphsTuple`
  """
  if node_features is not None:
    num_node_features = jax.tree_leaves(node_features)[0].shape[0]
    if n_node_per_graph * n_graph != num_node_features:
      raise ValueError(
          'Number of nodes is not equal to num_nodes_per_graph * n_graph.')
  if global_features is not None:
    if n_graph != jax.tree_leaves(global_features)[0].shape[0]:
      raise ValueError('The number of globals is not equal to n_graph.')
  senders = []
  receivers = []
  n_edge = []
  tmp_senders, tmp_receivers = jnp.meshgrid(
      jnp.arange(n_node_per_graph), jnp.arange(n_node_per_graph))
  if not add_self_edges:
    tmp_senders = jax.vmap(jnp.roll)(
        tmp_senders, -jnp.arange(len(tmp_senders)))[:, 1:]
    tmp_receivers = tmp_receivers[:, 1:]
  # Flatten the senders and receivers.
  tmp_senders = tmp_senders.flatten()
  tmp_receivers = tmp_receivers.flatten()
  for graph_idx in range(n_graph):
    offset = graph_idx * n_node_per_graph
    senders.append(tmp_senders + offset)
    receivers.append(tmp_receivers + offset)
    n_edge.append(len(tmp_senders))

  def _concat_or_empty_indices(indices_list):
    if indices_list:
      return jnp.concatenate(indices_list, axis=0)
    else:
      return jnp.array([], dtype=tmp_senders.dtype)
  return gn_graph.GraphsTuple(
      nodes=node_features,
      edges=None,
      n_node=jnp.array([n_node_per_graph]*n_graph),
      n_edge=jnp.array(n_edge) if n_edge else jnp.array([0]),
      senders=_concat_or_empty_indices(senders),
      receivers=_concat_or_empty_indices(receivers),
      globals=global_features,
  )

