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
"""A library of Graph Neural Network models."""

import functools
from typing import Any, Callable, Iterable, Mapping, Optional, Union

import jax
import jax.numpy as jnp
import jax.tree_util as tree
from jraph._src import graph as gn_graph
from jraph._src import utils

# As of 04/2020 pytype doesn't support recursive types.
# pytype: disable=not-supported-yet
ArrayTree = Union[jnp.ndarray, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]

# All features will be an ArrayTree.
NodeFeatures = EdgeFeatures = SenderFeatures = ReceiverFeatures = Globals = ArrayTree

# Signature:
# (edges of each node to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToNodesFn = Callable[
    [EdgeFeatures, jnp.ndarray, int], NodeFeatures]

# Signature:
# (nodes of each graph to be aggregated, segment ids, number of segments) ->
# aggregated nodes
AggregateNodesToGlobalsFn = Callable[[NodeFeatures, jnp.ndarray, int],
                                     Globals]

# Signature:
# (edges of each graph to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToGlobalsFn = Callable[[EdgeFeatures, jnp.ndarray, int],
                                     Globals]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# attention weights
AttentionLogitFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], ArrayTree]

# Signature:
# (edge features, weights) -> edge features for node update
AttentionReduceFn = Callable[[EdgeFeatures, ArrayTree], EdgeFeatures]

# Signature:
# (edges to be normalized, segment ids, number of segments) ->
# normalized edges
AttentionNormalizeFn = Callable[[EdgeFeatures, jnp.ndarray, int], EdgeFeatures]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# updated edge features
GNUpdateEdgeFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], EdgeFeatures]

# Signature:
# (node features, outgoing edge features, incoming edge features,
#  globals) -> updated node features
GNUpdateNodeFn = Callable[
    [NodeFeatures, SenderFeatures, ReceiverFeatures, Globals], NodeFeatures]

GNUpdateGlobalFn = Callable[[NodeFeatures, EdgeFeatures, Globals], Globals]


def GraphNetwork(
    update_edge_fn: Optional[GNUpdateEdgeFn],
    update_node_fn: Optional[GNUpdateNodeFn],
    update_global_fn: Optional[GNUpdateGlobalFn] = None,
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils
    .segment_sum,
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils
    .segment_sum,
    attention_logit_fn: Optional[AttentionLogitFn] = None,
    attention_normalize_fn: Optional[AttentionNormalizeFn] = utils
    .segment_softmax,
    attention_reduce_fn: Optional[AttentionReduceFn] = None):
  """Returns a method that applies a configured GraphNetwork.

  This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

  There is one difference. For the nodes update the class aggregates over the
  sender edges and receiver edges separately. This is a bit more general
  than the algorithm described in the paper. The original behaviour can be
  recovered by using only the receiver edge aggregations for the update.

  In addition this implementation supports softmax attention over incoming
  edge features.

  Example usage::

    gn = GraphNetwork(update_edge_function,
    update_node_function, **kwargs)
    # Conduct multiple rounds of message passing with the same parameters:
    for _ in range(num_message_passing_steps):
      graph = gn(graph)

  Args:
    update_edge_fn: function used to update the edges or None to deactivate edge
      updates.
    update_node_fn: function used to update the nodes or None to deactivate node
      updates.
    update_global_fn: function used to update the globals or None to deactivate
      globals updates.
    aggregate_edges_for_nodes_fn: function used to aggregate messages to each
      node.
    aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
      globals.
    aggregate_edges_for_globals_fn: function used to aggregate the edges for the
      globals.
    attention_logit_fn: function used to calculate the attention weights or
      None to deactivate attention mechanism.
    attention_normalize_fn: function used to normalize raw attention logits or
      None if attention mechanism is not active.
    attention_reduce_fn: function used to apply weights to the edge features or
      None if attention mechanism is not active.

  Returns:
    A method that applies the configured GraphNetwork.
  """
  not_both_supplied = lambda x, y: (x != y) and ((x is None) or (y is None))
  if not_both_supplied(attention_reduce_fn, attention_logit_fn):
    raise ValueError(('attention_logit_fn and attention_reduce_fn must both be'
                      ' supplied.'))

  def _ApplyGraphNet(graph):
    """Applies a configured GraphNetwork to a graph.

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
    nodes, edges, receivers, senders, globals_, n_node, n_edge = graph
    # Equivalent to jnp.sum(n_node), but jittable
    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
    sum_n_edge = senders.shape[0]
    if not tree.tree_all(
        tree.tree_map(lambda n: n.shape[0] == sum_n_node, nodes)):
      raise ValueError(
          'All node arrays in nest must contain the same number of nodes.')

    sent_attributes = tree.tree_map(lambda n: n[senders], nodes)
    received_attributes = tree.tree_map(lambda n: n[receivers], nodes)
    # Here we scatter the global features to the corresponding edges,
    # giving us tensors of shape [num_edges, global_feat].
    global_edge_attributes = tree.tree_map(lambda g: jnp.repeat(
        g, n_edge, axis=0, total_repeat_length=sum_n_edge), globals_)

    if update_edge_fn:
      edges = update_edge_fn(edges, sent_attributes, received_attributes,
                             global_edge_attributes)

    if attention_logit_fn:
      logits = attention_logit_fn(edges, sent_attributes, received_attributes,
                                  global_edge_attributes)
      tree_calculate_weights = functools.partial(
          attention_normalize_fn,
          segment_ids=receivers,
          num_segments=sum_n_node)
      weights = tree.tree_map(tree_calculate_weights, logits)
      edges = attention_reduce_fn(edges, weights)

    if update_node_fn:
      sent_attributes = tree.tree_map(
          lambda e: aggregate_edges_for_nodes_fn(e, senders, sum_n_node), edges)
      received_attributes = tree.tree_map(
          lambda e: aggregate_edges_for_nodes_fn(e, receivers, sum_n_node),
          edges)
      # Here we scatter the global features to the corresponding nodes,
      # giving us tensors of shape [num_nodes, global_feat].
      global_attributes = tree.tree_map(lambda g: jnp.repeat(
          g, n_node, axis=0, total_repeat_length=sum_n_node), globals_)
      nodes = update_node_fn(nodes, sent_attributes,
                             received_attributes, global_attributes)

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
          graph_idx, n_edge, axis=0, total_repeat_length=sum_n_edge)
      # We use the aggregation function to pool the nodes/edges per graph.
      node_attributes = tree.tree_map(
          lambda n: aggregate_nodes_for_globals_fn(n, node_gr_idx, n_graph),
          nodes)
      edge_attribtutes = tree.tree_map(
          lambda e: aggregate_edges_for_globals_fn(e, edge_gr_idx, n_graph),
          edges)
      # These pooled nodes are the inputs to the global update fn.
      globals_ = update_global_fn(node_attributes, edge_attribtutes, globals_)
    # pylint: enable=g-long-lambda
    return gn_graph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        receivers=receivers,
        senders=senders,
        globals=globals_,
        n_node=n_node,
        n_edge=n_edge)

  return _ApplyGraphNet


InteractionUpdateNodeFn = Callable[
    [NodeFeatures,
     Mapping[str, SenderFeatures],
     Mapping[str, ReceiverFeatures]],
    NodeFeatures]
InteractionUpdateNodeFnNoSentEdges = Callable[
    [NodeFeatures,
     Mapping[str, ReceiverFeatures]],
    NodeFeatures]

InteractionUpdateEdgeFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures], EdgeFeatures]


def InteractionNetwork(
    update_edge_fn: InteractionUpdateEdgeFn,
    update_node_fn: Union[InteractionUpdateNodeFn,
                          InteractionUpdateNodeFnNoSentEdges],
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    include_sent_messages_in_node_update: bool = False):
  """Returns a method that applies a configured InteractionNetwork.

  An interaction network computes interactions on the edges based on the
  previous edges features, and on the features of the nodes sending into those
  edges. It then updates the nodes based on the incoming updated edges.
  See https://arxiv.org/abs/1612.00222 for more details.

  This implementation adds an option not in https://arxiv.org/abs/1612.00222,
  which is to include edge features for which a node is a sender in the
  arguments to the node update function.

  Args:
    update_edge_fn: a function mapping a single edge update inputs to a single
      edge feature.
    update_node_fn: a function mapping a single node update input to a single
      node feature.
    aggregate_edges_for_nodes_fn: function used to aggregate messages to each
      node.
    include_sent_messages_in_node_update: pass edge features for which a node is
      a sender to the node update function.
  """
  # An InteractionNetwork is a GraphNetwork without globals features,
  # so we implement the InteractionNetwork as a configured GraphNetwork.

  # An InteractionNetwork edge function does not have global feature inputs,
  # so we filter the passed global argument in the GraphNetwork.
  wrapped_update_edge_fn = lambda e, s, r, g: update_edge_fn(e, s, r)

  # Similarly, we wrap the update_node_fn to ensure only the expected
  # arguments are passed to the Interaction net.
  if include_sent_messages_in_node_update:
    wrapped_update_node_fn = lambda n, s, r, g: update_node_fn(n, s, r)
  else:
    wrapped_update_node_fn = lambda n, s, r, g: update_node_fn(n, r)
  return GraphNetwork(
      update_edge_fn=wrapped_update_edge_fn,
      update_node_fn=wrapped_update_node_fn,
      aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn)


# Signature:
# edge features -> embedded edge features
EmbedEdgeFn = Callable[[EdgeFeatures], EdgeFeatures]

# Signature:
# node features -> embedded node features
EmbedNodeFn = Callable[[NodeFeatures], NodeFeatures]

# Signature:
# globals features -> embedded globals features
EmbedGlobalFn = Callable[[Globals], Globals]


def GraphMapFeatures(embed_edge_fn: Optional[EmbedEdgeFn] = None,
                     embed_node_fn: Optional[EmbedNodeFn] = None,
                     embed_global_fn: Optional[EmbedGlobalFn] = None):
  """Returns function which embeds the components of a graph independently.

  Args:
    embed_edge_fn: function used to embed the edges.
    embed_node_fn: function used to embed the nodes.
    embed_global_fn: function used to embed the globals.
  """
  identity = lambda x: x
  embed_edges_fn = embed_edge_fn if embed_edge_fn else identity
  embed_nodes_fn = embed_node_fn if embed_node_fn else identity
  embed_global_fn = embed_global_fn if embed_global_fn else identity

  def Embed(graphs_tuple):
    return graphs_tuple._replace(
        nodes=embed_nodes_fn(graphs_tuple.nodes),
        edges=embed_edges_fn(graphs_tuple.edges),
        globals=embed_global_fn(graphs_tuple.globals))

  return Embed


def RelationNetwork(
    update_edge_fn: Callable[[SenderFeatures, ReceiverFeatures], EdgeFeatures],
    update_global_fn: Callable[[EdgeFeatures], NodeFeatures],
    aggregate_edges_for_globals_fn:
        AggregateEdgesToGlobalsFn = utils.segment_sum):
  """Returns a method that applies a Relation Network.

  See https://arxiv.org/abs/1706.01427 for more details.

  This implementation has one more argument, `aggregate_edges_for_globals_fn`,
  which changes how edge features are aggregated. The paper uses the default -
  `utils.segment_sum`.

  Args:
    update_edge_fn: function used to update the edges.
    update_global_fn: function used to update the globals.
    aggregate_edges_for_globals_fn: function used to aggregate the edges for the
      globals.
  """
  return GraphNetwork(
      update_edge_fn=lambda e, s, r, g: update_edge_fn(s, r),
      update_node_fn=None,
      update_global_fn=lambda n, e, g: update_global_fn(e),
      attention_logit_fn=None,
      aggregate_edges_for_globals_fn=aggregate_edges_for_globals_fn)


def DeepSets(
    update_node_fn: Callable[[NodeFeatures, Globals], NodeFeatures],
    update_global_fn: Callable[[NodeFeatures], Globals],
    aggregate_nodes_for_globals_fn:
        AggregateNodesToGlobalsFn = utils.segment_sum):
  """Returns a method that applies a DeepSets layer.

  Implementation for the model described in https://arxiv.org/abs/1703.06114
  (M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov, A. Smola).
  See also PointNet (https://arxiv.org/abs/1612.00593, C. Qi, H. Su, K. Mo,
  L. J. Guibas) for a related model.

  This module operates on sets, which can be thought of as graphs without
  edges. The nodes features are first updated based on their value and the
  globals features, and new globals features are then computed based on the
  updated nodes features.

  Args:
    update_node_fn: function used to update the nodes.
    update_global_fn: function used to update the globals.
    aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
      globals.
  """
  # DeepSets can be implemented with a GraphNetwork, with just a node
  # update function that takes nodes and globals, and a global update
  # function based on the updated node features.
  return GraphNetwork(
      update_edge_fn=None,
      update_node_fn=lambda n, s, r, g: update_node_fn(n, g),
      update_global_fn=lambda n, e, g: update_global_fn(n),
      aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn)


def GraphNetGAT(
    update_edge_fn: GNUpdateEdgeFn,
    update_node_fn: GNUpdateNodeFn,
    attention_logit_fn: AttentionLogitFn,
    attention_reduce_fn: AttentionReduceFn,
    update_global_fn: Optional[GNUpdateGlobalFn] = None,
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils.
    segment_sum,
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils.
    segment_sum
    ):
  """Returns a method that applies a GraphNet with attention on edge features.

  Args:
    update_edge_fn: function used to update the edges.
    update_node_fn: function used to update the nodes.
    attention_logit_fn: function used to calculate the attention weights.
    attention_reduce_fn: function used to apply attention weights to the edge
      features.
    update_global_fn: function used to update the globals or None to deactivate
      globals updates.
    aggregate_edges_for_nodes_fn: function used to aggregate attention-weighted
      messages to each node.
    aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
      globals.
    aggregate_edges_for_globals_fn: function used to aggregate
      attention-weighted edges for the globals.

  Returns:
    A function that applies a GraphNet Graph Attention layer.
  """
  if (attention_logit_fn is None) or (attention_reduce_fn is None):
    raise ValueError(('`None` value not supported for `attention_logit_fn` or '
                      '`attention_reduce_fn` in a Graph Attention network.'))
  return GraphNetwork(
      update_edge_fn=update_edge_fn,
      update_node_fn=update_node_fn,
      update_global_fn=update_global_fn,
      attention_logit_fn=attention_logit_fn,
      attention_reduce_fn=attention_reduce_fn,
      aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
      aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn,
      aggregate_edges_for_globals_fn=aggregate_edges_for_globals_fn)


GATAttentionQueryFn = Callable[[NodeFeatures], NodeFeatures]
GATAttentionLogitFn = Callable[
    [SenderFeatures, ReceiverFeatures, EdgeFeatures], EdgeFeatures]
GATNodeUpdateFn = Callable[[NodeFeatures], NodeFeatures]


def GAT(attention_query_fn: GATAttentionQueryFn,
        attention_logit_fn: GATAttentionLogitFn,
        node_update_fn: Optional[GATNodeUpdateFn] = None):
  """Returns a method that applies a Graph Attention Network layer.

  Graph Attention message passing as described in
  https://arxiv.org/abs/1710.10903. This model expects node features as a
  jnp.array, may use edge features for computing attention weights, and
  ignore global features. It does not support nests.

  NOTE: this implementation assumes that the input graph has self edges. To
  recover the behavior of the referenced paper, please add self edges.

  Args:
    attention_query_fn: function that generates attention queries
      from sender node features.
    attention_logit_fn: function that converts attention queries into logits for
      softmax attention.
    node_update_fn: function that updates the aggregated messages. If None,
      will apply leaky relu and concatenate (if using multi-head attention).

  Returns:
    A function that applies a Graph Attention layer.
  """
  # pylint: disable=g-long-lambda
  if node_update_fn is None:
    # By default, apply the leaky relu and then concatenate the heads on the
    # feature axis.
    node_update_fn = lambda x: jnp.reshape(
        jax.nn.leaky_relu(x), (x.shape[0], -1))
  def _ApplyGAT(graph):
    """Applies a Graph Attention layer."""
    nodes, edges, receivers, senders, _, _, _ = graph
    # Equivalent to the sum of n_node, but statically known.
    try:
      sum_n_node = nodes.shape[0]
    except IndexError:
      raise IndexError('GAT requires node features')  # pylint: disable=raise-missing-from

    # First pass nodes through the node updater.
    nodes = attention_query_fn(nodes)
    # pylint: disable=g-long-lambda
    # We compute the softmax logits using a function that takes the
    # embedded sender and receiver attributes.
    sent_attributes = nodes[senders]
    received_attributes = nodes[receivers]
    softmax_logits = attention_logit_fn(
        sent_attributes, received_attributes, edges)

    # Compute the softmax weights on the entire tree.
    weights = utils.segment_softmax(softmax_logits, segment_ids=receivers,
                                    num_segments=sum_n_node)
    # Apply weights
    messages = sent_attributes * weights
    # Aggregate messages to nodes.
    nodes = utils.segment_sum(messages, receivers, num_segments=sum_n_node)

    # Apply an update function to the aggregated messages.
    nodes = node_update_fn(nodes)
    return graph._replace(nodes=nodes)
  # pylint: enable=g-long-lambda
  return _ApplyGAT


def GraphConvolution(
    update_node_fn: Callable[[NodeFeatures], NodeFeatures],
    aggregate_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    add_self_edges: bool = False,
    symmetric_normalization: bool = True):
  """Returns a method that applies a Graph Convolution layer.

  Graph Convolutional layer as in https://arxiv.org/abs/1609.02907,

  NOTE: This implementation does not add an activation after aggregation.
  If you are stacking layers, you may want to add an activation between
  each layer.

  Args:
    update_node_fn: function used to update the nodes. In the paper a single
      layer MLP is used.
    aggregate_nodes_fn: function used to aggregates the sender nodes.
    add_self_edges: whether to add self edges to nodes in the graph as in the
      paper definition of GCN. Defaults to False.
    symmetric_normalization: whether to use symmetric normalization. Defaults
      to True. Note that to replicate the fomula of the linked paper, the
      adjacency matrix must be symmetric. If the adjacency matrix is not
      symmetric the data is prenormalised by the sender degree matrix and post
      normalised by the receiver degree matrix.

  Returns:
    A method that applies a Graph Convolution layer.
  """
  def _ApplyGCN(graph):
    """Applies a Graph Convolution layer."""
    nodes, _, receivers, senders, _, _, _ = graph

    # First pass nodes through the node updater.
    nodes = update_node_fn(nodes)
    # Equivalent to jnp.sum(n_node), but jittable
    total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]
    if add_self_edges:
      # We add self edges to the senders and receivers so that each node
      # includes itself in aggregation.
      # In principle, a `GraphsTuple` should partition by n_edge, but in
      # this case it is not required since a GCN is agnostic to whether
      # the `GraphsTuple` is a batch of graphs or a single large graph.
      conv_receivers = jnp.concatenate((receivers, jnp.arange(total_num_nodes)),
                                       axis=0)
      conv_senders = jnp.concatenate((senders, jnp.arange(total_num_nodes)),
                                     axis=0)
    else:
      conv_senders = senders
      conv_receivers = receivers

    # pylint: disable=g-long-lambda
    if symmetric_normalization:
      # Calculate the normalization values.
      count_edges = lambda x: utils.segment_sum(
          jnp.ones_like(conv_senders), x, total_num_nodes)
      sender_degree = count_edges(conv_senders)
      receiver_degree = count_edges(conv_receivers)

      # Pre normalize by sqrt sender degree.
      # Avoid dividing by 0 by taking maximum of (degree, 1).
      nodes = tree.tree_map(
          lambda x: x * jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))[:, None],
          nodes,
      )
      # Aggregate the pre normalized nodes.
      nodes = tree.tree_map(
          lambda x: aggregate_nodes_fn(x[conv_senders], conv_receivers,
                                       total_num_nodes), nodes)
      # Post normalize by sqrt receiver degree.
      # Avoid dividing by 0 by taking maximum of (degree, 1).
      nodes = tree.tree_map(
          lambda x:
          (x * jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))[:, None]),
          nodes,
      )
    else:
      nodes = tree.tree_map(
          lambda x: aggregate_nodes_fn(x[conv_senders], conv_receivers,
                                       total_num_nodes), nodes)
    # pylint: enable=g-long-lambda
    return graph._replace(nodes=nodes)

  return _ApplyGCN
