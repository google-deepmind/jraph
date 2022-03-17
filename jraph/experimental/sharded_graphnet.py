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
"""Sharded (Data Parallel) Graph Nets."""

import functools
from typing import Callable, List, NamedTuple, Optional
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
from jraph._src import graph as gn_graph
from jraph._src import utils
import numpy as np


class ShardedEdgesGraphsTuple(NamedTuple):
  """A `GraphsTuple` for use with `ShardedEdgesGraphNetwork`.

  NOTES:
  - A ShardedEdgesGraphNetwork is for use with `jax.pmap`. As such, it will have
    a leading axis of size `num_devices` on the host, but no such axis on
    device. Non-sharded data is replicated on each device. To achieve this with
    `jax.pmap` you can broadcast non-sharded data to have leading axis
    'num_devices' or use the 'in_axes' parameter, which will indicate which
    attributes should be replicated and which should not. Current helper methods
    use the first approach.
  - It is recommended that you constructed ShardedEdgesGraphsTuples with
    `graphs_tuple_to_broadcasted_sharded_grahs_tuple`.


  The values of `nodes`, `device_edges` and `globals` can be gn_graph.ArrayTree
  - nests of features with `jax` compatible values. For example, `nodes` in a
  graph may have more than one type of attribute.

  However, the ShardedEdgesGraphsTuple typically takes the following form for a
  batch of `n` graphs:

  - n_node: The number of nodes per graph. It is a vector of integers with shape
    `[n_graphs]`, such that `graph.n_node[i]` is the number of nodes in the i-th
    graph.

  - n_edge: The number of edges per graph. It is a vector of integers with shape
    `[n_graphs]`, such that `graph.n_edge[i]` is the number of edges in the i-th
    graph.

  - nodes: The nodes features. It is either `None` (the graph has no node
    features), or a vector of shape `[n_nodes] + node_shape`, where
    `n_nodes = sum(graph.n_node)` is the total number of nodes in the batch of
    graphs, and `node_shape` represents the shape of the features of each node.
    The relative index of a node from the batched version can be recovered from
    the `graph.n_node` property. For instance, the second node of the third
    graph will have its features in the
    `1 + graph.n_node[0] + graph.n_node[1]`-th slot of graph.nodes.
    Observe that having a `None` value for this field does not mean that the
    graphs have no nodes, only that they do not have node features.

  - receivers: The indices of the receiver nodes, for each edge. It is either
    `None` (if the graph has no edges), or a vector of integers of shape
    `[n_edges]`, such that `graph.receivers[i]` is the index of the node
    receiving from the i-th edge.

    Observe that the index is absolute (in other words, cumulative), i.e.
    `graphs.receivers` take value in `[0, n_nodes]`. For instance, an edge
    connecting the vertices with relative indices 2 and 3 in the second graph of
    the batch would have a `receivers` value of `3 + graph.n_node[0]`.
    If `graphs.receivers` is `None`, then `graphs.edges` and `graphs.senders`
    should also be `None`.

  - senders: The indices of the sender nodes, for each edge. It is either
    `None` (if the graph has no edges), or a vector of integers of shape
    `[n_edges]`, such that `graph.senders[i]` is the index of the node
    sending from the i-th edge.

    Observe that the index is absolute, i.e. `graphs.senders` take value in
    `[0, n_nodes]`. For instance, an edge connecting the vertices with relative
    indices 1 and 3 in the third graph of the batch would have a `senders` value
    of `1 + graph.n_node[0] + graph.n_node[1]`.

    If `graphs.senders` is `None`, then `graphs.edges` and `graphs.receivers`
    should also be `None`.

  - globals: The global features of the graph. It is either `None` (the graph
    has no global features), or a vector of shape `[n_graphs] + global_shape`
    representing graph level features.

  The ShardedEdgesGraphsTuple also contains device-local attributes that are
  used for data parallel computation. On the host, each of these attributes will
  have an additional leading axis of shape `num_devices` for use with
  `jax.pmap`, but this is ommited in the following documentation.

  - device_edges: The subset of the edge features that are on the device.
      It is either `None` (the graph has no edge features), or a vector of
      shape `[num_edges / num_devices] + edge_shape`

      Observe that having a `None` value for this field does not mean that the
      graph has no edges, only that they do not have edge features.

  - device_senders: The sender indices of edges on device. This is of length
      num_edges / num_devices.

  - device_receivers: The receiver indices of edge on device. This is of length
      num_edges / num_devices.

  - device_n_edge: The graph partitions of the edges on device. For example,
      say that there are 2 graphs in the original graphs tuple, with n_edge
      [1, 11], which has been split over 3 devices. The `device_n_edge`s would
      be [[1, 3], [4, 0], [4, 0]]. `0` valued entries that are padding values or
      graphs with zero edges are not distinguished. Since these attributes are
      used only for `repeat` purposes, the difference makes no difference to
      the implementation.

  - device_graph_idx: The indices of the graphs on device. For example, say
      that there are 5 graphs in the original graphs tuple, and these has been
      split over 3 devices, the device_graphs_idxs could be
      [[0, 1, 2], [2, 3, 0], [3, 4, 0]]. In this splitting, the third graph
      is split over 2 devices. If a `0` is the first in `device_graph_idx` then
      that indicates the first graph, otherwise it indicates a padding value.
  """
  nodes: gn_graph.ArrayTree
  device_edges: gn_graph.ArrayTree
  device_receivers: jnp.ndarray  # with integer dtype
  device_senders: jnp.ndarray  # with integer dtype
  receivers: jnp.ndarray  # with integer dtype
  senders: jnp.ndarray  # with integer dtype
  globals: gn_graph.ArrayTree
  device_n_edge: jnp.ndarray  # with integer dtype
  n_node: jnp.ndarray  # with integer dtype
  n_edge: jnp.ndarray  # with integer dtype
  device_graph_idx: jnp.ndarray  # with integer dtype


def graphs_tuple_to_broadcasted_sharded_graphs_tuple(
    graphs_tuple: jraph.GraphsTuple,
    num_shards: int) -> ShardedEdgesGraphsTuple:
  """Converts a `GraphsTuple` to a `ShardedEdgesGraphsTuple` to use with `pmap`.

  For a given number of shards this will compute device-local edge and graph
  attributes, and add a batch axis of size num_shards. You can then use
  `ShardedEdgesGraphNetwork` with `jax.pmap`.

  Args:
    graphs_tuple: The `GraphsTuple` to be converted to a sharded `GraphsTuple`.
    num_shards: The number of devices to shard over.

  Returns:
    A ShardedEdgesGraphsTuple over the number of shards.
  """
  # Note: this is not jittable, so to prevent using a device by accident,
  # this is all happening in numpy.
  nodes, edges, receivers, senders, globals_, n_node, n_edge = graphs_tuple
  if np.sum(n_edge) % num_shards != 0:
    raise ValueError(('The number of edges in a `graph.GraphsTuple` must be '
                      'divisible by the number of devices per replica.'))
  if np.sum(np.array(n_edge)) == 0:
    raise ValueError('The input `Graphstuple` must have edges.')
  # Broadcast replicated features to have a `num_shards` leading axis.
  # pylint: disable=g-long-lambda
  broadcast = lambda x: np.broadcast_to(x[None, :], (num_shards,) + x.shape)
  # pylint: enable=g-long-lambda

  # `edges` will be straightforwardly sharded, with 1/num_shards of
  # the edges on each device.
  def shard_edges(edge_features):
    return np.reshape(edge_features, (num_shards, -1) + edge_features.shape[1:])

  edges = jax.tree_map(shard_edges, edges)
  # Our sharded strategy is by edges - which means we need a device local
  # n_edge, senders and receivers to do global aggregations.

  # Senders and receivers are easy - 1/num_shards per device.
  device_senders = shard_edges(senders)
  device_receivers = shard_edges(receivers)

  # n_edge is a bit more difficult. Let's say we have a graphs tuple with
  # n_edge [2, 8], and we want to distribute this on two devices. Then
  # we will have sharded the edges to [5, 5], so the n_edge per device will be
  # [2,3], and [5]. Since we need to have each of the n_edge the same shape,
  # we will need to pad this to [5,0]. This is a bit dangerous, as the zero
  # here has a different meaning to a graph with zero edges, but we need the
  # zero for the global broadcasting to be correct for aggregation. Since
  # this will only be used in the first instance for global broadcasting on
  # device I think this is ok, but ideally we'd have a more elegant solution.
  # TODO(jonathangodwin): think of a more elegant solution.
  edges_per_device = np.sum(n_edge) // num_shards
  edges_in_current_split = 0
  completed_splits = []
  current_split = {'n_edge': [], 'device_graph_idx': []}
  for device_graph_idx, x in enumerate(n_edge):
    new_edges_in_current_split = edges_in_current_split + x
    if new_edges_in_current_split > edges_per_device:
      # A single graph may be spread across multiple replicas, so here we
      # iteratively create new splits until the graph is exhausted.

      # How many edges we are trying to allocate.
      carry = x
      # How much room there is in the current split for new edges.
      space_in_current_split = edges_per_device - edges_in_current_split
      while carry > 0:
        if carry >= space_in_current_split:
          # We've encountered a situation where we need to split a graph across
          # >= 2 devices. We compute the number we will carry to the next split,
          # and add a full split.
          carry = carry - space_in_current_split
          # Add the left edges to the current split, and complete the split
          # by adding it to completed_splits.
          current_split['n_edge'].append(space_in_current_split)
          current_split['device_graph_idx'].append(device_graph_idx)
          completed_splits.append(current_split)
          # reset the split
          current_split = {'n_edge': [], 'device_graph_idx': []}

          space_in_current_split = edges_per_device
          edges_in_current_split = 0
        else:
          current_split = {
              'n_edge': [carry],
              'device_graph_idx': [device_graph_idx]
          }
          edges_in_current_split = carry
          carry = 0
          # Since the total number of edges must be divisible by the number
          # of devices, this code  path can only be executed for an intermediate
          # graph, thus it is not a complete split and we never need to add it
          # to `completed splits`.
    else:
      # Add the edges and globals to the current split.
      current_split['n_edge'].append(x)
      current_split['device_graph_idx'].append(device_graph_idx)
      # If we've reached the end of a split, complete it and start a new one.
      if new_edges_in_current_split == edges_per_device:
        completed_splits.append(current_split)
        current_split = {'n_edge': [], 'device_graph_idx': []}
        edges_in_current_split = 0
      else:
        edges_in_current_split = new_edges_in_current_split

  # Flatten list of dicts to dict of lists.
  completed_splits = {
      k: [d[k] for d in completed_splits] for k in completed_splits[0]
  }
  pad_split_to = max([len(x) for x in completed_splits['n_edge']])
  pad = lambda x: np.pad(x, (0, pad_split_to - len(x)), mode='constant')
  device_n_edge = np.array([pad(x) for x in completed_splits['n_edge']])
  device_graph_idx = np.array(
      [pad(x) for x in completed_splits['device_graph_idx']])
  return ShardedEdgesGraphsTuple(
      nodes=jax.tree_map(broadcast, nodes),
      device_edges=edges,
      device_receivers=device_receivers,
      device_senders=device_senders,
      receivers=broadcast(receivers),
      senders=broadcast(senders),
      device_graph_idx=device_graph_idx,
      globals=jax.tree_map(broadcast, globals_),
      n_node=broadcast(n_node),
      n_edge=broadcast(n_edge),
      device_n_edge=device_n_edge)


def broadcasted_sharded_graphs_tuple_to_graphs_tuple(sharded_graphs_tuple):
  """Converts a broadcasted ShardedGraphsTuple to a GraphsTuple."""
  # We index the first element of replicated arrays, since they have been
  # repeated. For edges, we reshape to recover all of the edge features.
  unbroadcast = lambda y: tree.tree_map(lambda x: x[0], y)
  unshard = lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:])
  # TODO(jonathangodwin): check senders and receivers are consistent.
  return jraph.GraphsTuple(
      nodes=unbroadcast(sharded_graphs_tuple.nodes),
      edges=tree.tree_map(unshard, sharded_graphs_tuple.device_edges),
      n_node=sharded_graphs_tuple.n_node[0],
      n_edge=sharded_graphs_tuple.n_edge[0],
      globals=unbroadcast(sharded_graphs_tuple.globals),
      senders=sharded_graphs_tuple.senders[0],
      receivers=sharded_graphs_tuple.receivers[0])


def sharded_segment_sum(data, indices, num_segments, axis_index_groups):
  """Segment sum over data on multiple devices."""
  device_segment_sum = utils.segment_sum(data, indices, num_segments)
  return jax.lax.psum(
      device_segment_sum, axis_name='i', axis_index_groups=axis_index_groups)


ShardedEdgeFeatures = gn_graph.ArrayTree
AggregateShardedEdgesToGlobalsFn = Callable[
    [ShardedEdgeFeatures, jnp.ndarray, int, jnp.ndarray], gn_graph.ArrayTree]
AggregateShardedEdgesToNodesFn = Callable[
    [gn_graph.ArrayTree, jnp.ndarray, int, List[List[int]]], jraph.NodeFeatures]


# pylint: disable=invalid-name
def ShardedEdgesGraphNetwork(
    update_edge_fn: Optional[jraph.GNUpdateEdgeFn],
    update_node_fn: Optional[jraph.GNUpdateNodeFn],
    update_global_fn: Optional[jraph.GNUpdateGlobalFn] = None,
    aggregate_edges_for_nodes_fn:
    AggregateShardedEdgesToNodesFn = sharded_segment_sum,
    aggregate_nodes_for_globals_fn: jraph.AggregateNodesToGlobalsFn = jax.ops
    .segment_sum,
    aggregate_edges_for_globals_fn:
    AggregateShardedEdgesToGlobalsFn = sharded_segment_sum,
    attention_logit_fn: Optional[jraph.AttentionLogitFn] = None,
    attention_reduce_fn: Optional[jraph.AttentionReduceFn] = None,
    num_shards: int = 1):
  """Returns a method that applies a GraphNetwork on a sharded GraphsTuple.

  This GraphNetwork is sharded over `edges`, all other features are assumed
  to be replicated on device.
  There are two clear use cases for a ShardedEdgesGraphNetwork. The first is
  where a single graph can't fit on device. The second is when you are compute
  bound on the edge feature calculation, and you'd like to speed up
  training/inference by distributing the compute across devices.

  Example usage:

    ```
    gn = jax.pmap(ShardedEdgesGraphNetwork(update_edge_function,
    update_node_function, **kwargs), axis_name='i')
    # Conduct multiple rounds of message passing with the same parameters:
    for _ in range(num_message_passing_steps):
      sharded_graph = gn(sharded_graph)
    ```

  Args:
    update_edge_fn: function used to update the edges or None to deactivate edge
      updates.
    update_node_fn: function used to update the nodes or None to deactivate node
      updates.
    update_global_fn: function used to update the globals or None to deactivate
      globals updates.
    aggregate_edges_for_nodes_fn: function used to aggregate messages to each
      nodes. This must support cross-device aggregations.
    aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
      globals.
    aggregate_edges_for_globals_fn: function used to aggregate the edges for the
      globals. This must support cross-device aggregations.
    attention_logit_fn: function used to calculate the attention weights or None
      to deactivate attention mechanism.
    attention_reduce_fn: function used to apply weights to the edge features or
      None if attention mechanism is not active.
    num_shards: how many devices per replica for sharding.

  Returns:
    A method that applies the configured GraphNetwork.
  """
  not_both_supplied = lambda x, y: (x != y) and ((x is None) or (y is None))
  if not_both_supplied(attention_reduce_fn, attention_logit_fn):
    raise ValueError(('attention_logit_fn and attention_reduce_fn must both be'
                      ' supplied.'))

  devices = jax.devices()
  num_devices = len(devices)
  assert num_devices % num_shards == 0
  num_replicas = num_devices // num_shards
  # The IDs within a replica.
  replica_ids = list(range(num_devices))
  # How the devices are grouped per replica.
  axis_groups = [
      replica_ids[i * num_shards:(i + 1) * num_shards]
      for i in range(num_replicas)
  ]

  def _ApplyGraphNet(graph: ShardedEdgesGraphsTuple) -> ShardedEdgesGraphsTuple:
    """Applies a configured GraphNetwork to a sharded graph.

    This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

    There is one difference. For the nodes update the class aggregates over the
    sender edges and receiver edges separately. This is a bit more general
    the algorithm described in the paper. The original behaviour can be
    recovered by using only the receiver edge aggregations for the update.

    In addition this implementation supports softmax attention over incoming
    edge features.


    Many popular Graph Neural Networks can be implemented as special cases of
    GraphNets, for more information please see the paper.

    Args:
      graph: a `GraphsTuple` containing the graph.

    Returns:
      Updated `GraphsTuple`.
    """
    # pylint: disable=g-long-lambda
    nodes, device_edges, device_receivers, device_senders, receivers, senders, globals_, device_n_edge, n_node, n_edge, device_graph_idx = graph
    # Equivalent to jnp.sum(n_node), but jittable.
    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
    sum_device_n_edge = device_senders.shape[0]
    if not tree.tree_all(
        tree.tree_map(lambda n: n.shape[0] == sum_n_node, nodes)):
      raise ValueError(
          'All node arrays in nest must contain the same number of nodes.')

    sent_attributes = tree.tree_map(lambda n: n[device_senders], nodes)
    received_attributes = tree.tree_map(lambda n: n[device_receivers], nodes)
    # Here we scatter the global features to the corresponding edges,
    # giving us tensors of shape [num_edges, global_feat].
    global_edge_attributes = tree.tree_map(
        lambda g: jnp.repeat(
            g[device_graph_idx], device_n_edge, axis=0,
            total_repeat_length=sum_device_n_edge),
        globals_)

    if update_edge_fn:
      device_edges = update_edge_fn(device_edges, sent_attributes,
                                    received_attributes, global_edge_attributes)

    if attention_logit_fn:
      logits = attention_logit_fn(device_edges, sent_attributes,
                                  received_attributes, global_edge_attributes)
      tree_calculate_weights = functools.partial(
          utils.segment_softmax, segment_ids=receivers, num_segments=sum_n_node)
      weights = tree.tree_map(tree_calculate_weights, logits)
      device_edges = attention_reduce_fn(device_edges, weights)

    if update_node_fn:
      # Aggregations over nodes are assumed to take place over devices
      # specified by the axis_groups (e.g. with sharded_segment_sum).
      sent_attributes = tree.tree_map(
          lambda e: aggregate_edges_for_nodes_fn(e, device_senders, sum_n_node,
                                                 axis_groups), device_edges)
      received_attributes = tree.tree_map(
          lambda e: aggregate_edges_for_nodes_fn(
              e, device_receivers, sum_n_node, axis_groups), device_edges)
      # Here we scatter the global features to the corresponding nodes,
      # giving us tensors of shape [num_nodes, global_feat].
      global_attributes = tree.tree_map(
          lambda g: jnp.repeat(
              g, n_node, axis=0, total_repeat_length=sum_n_node), globals_)
      nodes = update_node_fn(nodes, sent_attributes, received_attributes,
                             global_attributes)

    if update_global_fn:
      n_graph = n_node.shape[0]
      graph_idx = jnp.arange(n_graph)
      # To aggregate nodes and edges from each graph to global features,
      # we first construct tensors that map the node to the corresponding graph.
      # For example, if you have `n_node=[1,2]`, we construct the tensor
      # [0, 1, 1]. We then do the same for edges.
      node_gr_idx = jnp.repeat(
          graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
      edge_gr_idx = jnp.repeat(
          device_graph_idx,
          device_n_edge,
          axis=0,
          total_repeat_length=sum_device_n_edge)
      # We use the aggregation function to pool the nodes/edges per graph.
      node_attributes = tree.tree_map(
          lambda n: aggregate_nodes_for_globals_fn(n, node_gr_idx, n_graph),
          nodes)
      edge_attribtutes = tree.tree_map(
          lambda e: aggregate_edges_for_globals_fn(e, edge_gr_idx, n_graph,
                                                   axis_groups), device_edges)
      # These pooled nodes are the inputs to the global update fn.
      globals_ = update_global_fn(node_attributes, edge_attribtutes, globals_)
    # pylint: enable=g-long-lambda
    return ShardedEdgesGraphsTuple(
        nodes=nodes,
        device_edges=device_edges,
        device_senders=device_senders,
        device_receivers=device_receivers,
        receivers=receivers,
        senders=senders,
        device_graph_idx=device_graph_idx,
        globals=globals_,
        n_node=n_node,
        n_edge=n_edge,
        device_n_edge=device_n_edge)

  return _ApplyGraphNet
# pylint: enable=invalid-name
