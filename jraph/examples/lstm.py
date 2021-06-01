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
"""Example of how to use recurrent networks (e.g.`LSTM`s) with `GraphNetwork`s.

Models can use the mechanism for specifying nested node, edge, or global
features to simultaneously keep inputs/embeddings together with a per-node,
per-edge or per-graph recurrent state.

In this example we show an `InteractionNetwork` that uses an LSTM to keep a
memory of the inputs to the edge model at each step of message passing, by using
separate "embedding" and "state" fields in the edge features.
Following a similar procedure, an LSTM could be added to the `node_update_fn`,
or even the `global_update_fn`, if using a full `GraphNetwork`.

Note it is recommended to use immutable container types to store nested edge,
node and global features to avoid unwanted side effects. In this example we
use `namedtuple`s.

"""

import collections

from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
import numpy as np


NUM_NODES = 5
NUM_EDGES = 7
NUM_MESSAGE_PASSING_STEPS = 10
EMBEDDING_SIZE = 32
HIDDEN_SIZE = 128

# Immutable class for storing nested node/edge features containing an embedding
# and a recurrent state.
StatefulField = collections.namedtuple("StatefulField", ["embedding", "state"])


def get_random_graph() -> jraph.GraphsTuple:
  return jraph.GraphsTuple(
      n_node=np.asarray([NUM_NODES]),
      n_edge=np.asarray([NUM_EDGES]),
      nodes=np.random.normal(size=[NUM_NODES, EMBEDDING_SIZE]),
      edges=np.random.normal(size=[NUM_EDGES, EMBEDDING_SIZE]),
      globals=None,
      senders=np.random.randint(0, NUM_NODES, [NUM_EDGES]),
      receivers=np.random.randint(0, NUM_NODES, [NUM_EDGES]))


def network_definition(graph: jraph.GraphsTuple) -> jraph.ArrayTree:
  """`InteractionNetwork` with an LSTM in the edge update."""

  # LSTM that will keep a memory of the inputs to the edge model.
  edge_fn_lstm = hk.LSTM(hidden_size=HIDDEN_SIZE)

  # MLPs used in the edge and the node model. Note that in this instance
  # the output size matches the input size so the same model can be run
  # iteratively multiple times. In a real model, this would usually be achieved
  # by first using an encoder in the input data into a common `EMBEDDING_SIZE`.
  edge_fn_mlp = hk.nets.MLP([HIDDEN_SIZE, EMBEDDING_SIZE])
  node_fn_mlp = hk.nets.MLP([HIDDEN_SIZE, EMBEDDING_SIZE])

  # Initialize the edge features to contain both the input edge embedding
  # and initial LSTM state. Note for the nodes we only have an embedding since
  # in this example nodes do not use a `node_fn_lstm`, but for analogy, we
  # still put it in a `StatefulField`.
  graph = graph._replace(
      edges=StatefulField(
          embedding=graph.edges,
          state=edge_fn_lstm.initial_state(graph.edges.shape[0])),
      nodes=StatefulField(embedding=graph.nodes, state=None),
  )

  def update_edge_fn(edges, sender_nodes, receiver_nodes):
    # We will run an LSTM memory on the inputs first, and then
    # process the output of the LSTM with an MLP.
    edge_inputs = jnp.concatenate([edges.embedding,
                                   sender_nodes.embedding,
                                   receiver_nodes.embedding], axis=-1)
    lstm_output, updated_state = edge_fn_lstm(edge_inputs, edges.state)
    updated_edges = StatefulField(
        embedding=edge_fn_mlp(lstm_output), state=updated_state,
    )
    return updated_edges

  def update_node_fn(nodes, received_edges):
    # Note `received_edges.state` will also contain the aggregated state for
    # all received edges, which we may choose to use in the node update.
    node_inputs = jnp.concatenate(
        [nodes.embedding, received_edges.embedding], axis=-1)
    updated_nodes = StatefulField(
        embedding=node_fn_mlp(node_inputs),
        state=None)
    return updated_nodes

  recurrent_graph_network = jraph.InteractionNetwork(
      update_edge_fn=update_edge_fn,
      update_node_fn=update_node_fn)

  # Apply the model recurrently for 10 message passing steps.
  # If instead we intended to use the LSTM to process a sequence of features
  # for each node/edge, here we would select the corresponding inputs from the
  # sequence along the sequence axis of the nodes/edges features to build the
  # correct input graph for each step of the iteration.
  num_message_passing_steps = 10
  for _ in range(num_message_passing_steps):
    graph = recurrent_graph_network(graph)

  return graph


def main(_):

  network = hk.without_apply_rng(hk.transform(network_definition))
  input_graph = get_random_graph()
  params = network.init(jax.random.PRNGKey(42), input_graph)
  output_graph = network.apply(params, input_graph)
  print(tree.tree_map(lambda x: x.shape, output_graph))


if __name__ == "__main__":
  app.run(main)
