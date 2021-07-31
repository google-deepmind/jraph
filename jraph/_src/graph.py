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
"""Graph Data Structures."""

from typing import Any, NamedTuple, Iterable, Mapping, Union, Optional
import jax.numpy as jnp


# As of 04/2020 pytype doesn't support recursive types.
# pytype: disable=not-supported-yet
ArrayTree = Union[jnp.ndarray, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]


class GraphsTuple(NamedTuple):
  """An ordered collection of graphs in a sparse format.

  The values of ``nodes``, ``edges`` and ``globals`` can be ``ArrayTrees`` -
  nests of features with ``jax`` compatible values. For example, ``nodes`` in a
  graph may have more than one type of attribute.

  However, the GraphsTuple typically takes the following form for a batch of
  `n` graphs:

  - n_node: The number of nodes per graph. It is a vector of integers with shape
    `[n_graphs]`, such that ``graph.n_node[i]`` is the number of nodes in the
    i-th graph.

  - n_edge: The number of edges per graph. It is a vector of integers with shape
    `[n_graphs]`, such that ``graph.n_edge[i]`` is the number of edges in the
    i-th graph.

  - nodes: The nodes features. It is either ``None`` (the graph has no node
    features), or a vector of shape `[n_nodes] + node_shape`, where
    ``n_nodes = sum(graph.n_node)`` is the total number of nodes in the batch of
    graphs, and `node_shape` represents the shape of the features of each node.
    The relative index of a node from the batched version can be recovered from
    the ``graph.n_node`` property. For instance, the second node of the third
    graph will have its features in the
    `1 + graph.n_node[0] + graph.n_node[1]`-th slot of graph.nodes.
    Observe that having a ``None`` value for this field does not mean that the
    graphs have no nodes, only that they do not have node features.

  - edges: The edges features. It is either ``None`` (the graph has no edge
    features), or a vector of shape `[n_edges] + edge_shape`, where
    ``n_edges = sum(graph.n_edge)`` is the total number of edges in the batch of
    graphs, and ``edge_shape`` represents the shape of the features of each
    edge.

    The relative index of an edge from the batched version can be recovered from
    the ``graph.n_edge`` property. For instance, the third edge of the third
    graph will have its features in the `2 + graph.n_edge[0] + graph.n_edge[1]`-
    th slot of graph.edges.

    Having a ``None`` value for this field does not necessarily mean that the
    graph has no edges, only that they do not have edge features.

  - receivers: The indices of the receiver nodes, for each edge. It is either
    ``None`` (if the graph has no edges), or a vector of integers of shape
    `[n_edges]`, such that ``graph.receivers[i]`` is the index of the node
    receiving from the i-th edge.

    Observe that the index is absolute (in other words, cumulative), i.e.
    ``graphs.receivers`` take value in `[0, n_nodes]`. For instance, an edge
    connecting the vertices with relative indices 2 and 3 in the second graph of
    the batch would have a ``receivers`` value of `3 + graph.n_node[0]`.
    If `graphs.receivers` is ``None``, then ``graphs.edges`` and
    ``graphs.senders`` should also be ``None``.

  - senders: The indices of the sender nodes, for each edge. It is either
    ``None`` (if the graph has no edges), or a vector of integers of shape
    `[n_edges]`, such that ``graph.senders[i]`` is the index of the node
    sending from the i-th edge.

    Observe that the index is absolute, i.e. ``graphs.senders`` take value in
    `[0, n_nodes]`. For instance, an edge connecting the vertices with relative
    indices 1 and 3 in the third graph of the batch would have a ``senders``
    value of `1 + graph.n_node[0] + graph.n_node[1]`.

    If ``graphs.senders`` is ``None``, then ``graphs.edges`` and
    ``graphs.receivers`` should also be ``None``.

  - globals: The global features of the graph. It is either ``None`` (the graph
    has no global features), or a vector of shape `[n_graphs] + global_shape`
    representing graph level features.


  """
  nodes: Optional[ArrayTree]
  edges: Optional[ArrayTree]
  receivers: Optional[jnp.ndarray]  # with integer dtype
  senders: Optional[jnp.ndarray]  # with integer dtype
  globals: Optional[ArrayTree]
  n_node: jnp.ndarray  # with integer dtype
  n_edge: jnp.ndarray   # with integer dtype
