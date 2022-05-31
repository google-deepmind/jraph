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
r"""Example training script for training OGB molhiv with jax graph-nets & flax.

The ogbg-molhiv dataset is a molecular property prediction dataset.
It is adopted from the MoleculeNet [1]. All the molecules are pre-processed
using RDKit [2].

Each graph represents a molecule, where nodes are atoms, and edges are chemical
bonds. Input node features are 9-dimensional, containing atomic number and
chirality, as well as other additional atom features such as formal charge and
whether the atom is in the ring or not.

The goal is to predict whether a molecule inhibits HIV virus replication or not.
Performance is measured in ROC-AUC.

This script uses a GraphNet to learn the prediction task.

[1] Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes,
Caleb Geniesse, Aneesh SPappu, Karl Leswing, and Vijay Pande.
Moleculenet: a benchmark for molecular machine learning.
Chemical Science, 9(2):513–530, 2018.

[2] Greg Landrum et al. RDKit: Open-source cheminformatics, 2006.

Example usage:

python3 train.py --data_path={DATA_PATH} --master_csv_path={MASTER_CSV_PATH} \
--save_dir={SAVE_DIR} --split_path={SPLIT_PATH}
"""


import functools
import logging
import pathlib
import pickle
from typing import Sequence
from absl import app
from absl import flags
from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
import jraph
from jraph.ogb_examples import data_utils


flags.DEFINE_string('data_path', None, 'Directory of the data.')
flags.DEFINE_string('split_path', None, 'Path to the data split indices.')
flags.DEFINE_string('master_csv_path', None, 'Path to OGB master.csv.')
flags.DEFINE_string('save_dir', None, 'Directory to save parameters to.')
flags.DEFINE_integer('batch_size', 1, 'Number of graphs in batch.')
flags.DEFINE_integer('num_training_steps', 1000, 'Number of training steps.')
flags.DEFINE_enum('mode', 'train', ['train', 'evaluate'], 'Train or evaluate.')
FLAGS = flags.FLAGS


class ExplicitMLP(nn.Module):
  """A flax MLP."""
  features: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, lyr in enumerate([nn.Dense(feat) for feat in self.features]):
      x = lyr(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
    return x


# Functions must be passed to jraph GNNs, but pytype does not recognise
# linen Modules as callables to here we wrap in a function.
def make_embed_fn(latent_size):
  def embed(inputs):
    return nn.Dense(latent_size)(inputs)
  return embed


def make_mlp(features):
  @jraph.concatenated_args
  def update_fn(inputs):
    return ExplicitMLP(features)(inputs)
  return update_fn


class GraphNetwork(nn.Module):
  """A flax GraphNetwork."""
  mlp_features: Sequence[int]
  latent_size: int

  @nn.compact
  def __call__(self, graph):
    # Add a global parameter for graph classification.
    graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))

    embedder = jraph.GraphMapFeatures(
        embed_node_fn=make_embed_fn(self.latent_size),
        embed_edge_fn=make_embed_fn(self.latent_size),
        embed_global_fn=make_embed_fn(self.latent_size))
    net = jraph.GraphNetwork(
        update_node_fn=make_mlp(self.mlp_features),
        update_edge_fn=make_mlp(self.mlp_features),
        # The global update outputs size 2 for binary classification.
        update_global_fn=make_mlp(self.mlp_features + (2,)))  # pytype: disable=unsupported-operands
    return net(embedder(graph))


def _nearest_bigger_power_of_two(x: int) -> int:
  """Computes the nearest power of two greater than x for padding."""
  y = 2
  while y < x:
    y *= 2
  return y


def pad_graph_to_nearest_power_of_two(
    graphs_tuple: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """Pads a batched `GraphsTuple` to the nearest power of two.

  For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
  would pad the `GraphsTuple` nodes and edges:
    7 nodes --> 8 nodes (2^3)
    5 edges --> 8 edges (2^3)

  And since padding is accomplished using `jraph.pad_with_graphs`, an extra
  graph and node is added:
    8 nodes --> 9 nodes
    3 graphs --> 4 graphs

  Args:
    graphs_tuple: a batched `GraphsTuple` (can be batch size 1).

  Returns:
    A graphs_tuple batched to the nearest power of two.
  """
  # Add 1 since we need at least one padding node for pad_with_graphs.
  pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_node)) + 1
  pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))
  # Add 1 since we need at least one padding graph for pad_with_graphs.
  # We do not pad to nearest power of two because the batch size is fixed.
  pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
  return jraph.pad_with_graphs(graphs_tuple, pad_nodes_to, pad_edges_to,
                               pad_graphs_to)


def compute_loss(params, graph, label, net):
  """Computes loss."""
  pred_graph = net.apply(params, graph)
  preds = jax.nn.log_softmax(pred_graph.globals)
  targets = jax.nn.one_hot(label, 2)

  # Since we have an extra 'dummy' graph in our batch due to padding, we want
  # to mask out any loss associated with the dummy graph.
  # Since we padded with `pad_with_graphs` we can recover the mask by using
  # get_graph_padding_mask.
  mask = jraph.get_graph_padding_mask(pred_graph)

  # Cross entropy loss.
  loss = -jnp.mean(preds * targets * mask[:, None])

  # Accuracy taking into account the mask.
  accuracy = jnp.sum(
      (jnp.argmax(pred_graph.globals, axis=1) == label) * mask)/jnp.sum(mask)
  return loss, accuracy


def train_step(optimizer, graph, label, net):
  partial_loss_fn = functools.partial(
      compute_loss, graph=graph, label=label, net=net)
  grad_fn = jax.value_and_grad(partial_loss_fn, has_aux=True)
  (loss, accuracy), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, {'loss': loss, 'accuracy': accuracy}


def train(data_path, master_csv_path, split_path, batch_size,
          num_training_steps, save_dir):
  """OGB Training Script."""

  # Initialize the dataset reader.
  reader = data_utils.DataReader(
      data_path=data_path,
      master_csv_path=master_csv_path,
      split_path=split_path,
      batch_size=batch_size)
  # Repeat the dataset forever for training.
  reader.repeat()

  net = GraphNetwork(mlp_features=(128, 128), latent_size=128)

  # Get a candidate graph and label to initialize the network.
  graph = reader.get_graph_by_idx(0)

  # Initialize the network.
  logging.info('Initializing network.')
  params = net.init(jax.random.PRNGKey(42), graph)
  optimizer = optim.Adam(learning_rate=1e-4).create(params)
  optimizer = jax.device_put(optimizer)

  for idx in range(num_training_steps):
    graph = next(reader)
    # Jax will re-jit your graphnet every time a new graph shape is encountered.
    # In the limit, this means a new compilation every training step, which
    # will result in *extremely* slow training. To prevent this, pad each
    # batch of graphs to the nearest power of two. Since jax maintains a cache
    # of compiled programs, the compilation cost is amortized.
    graph = pad_graph_to_nearest_power_of_two(graph)

    # Remove the label from the input graph/
    label = graph.globals['label']
    graph = graph._replace(globals={})

    optimizer, scalars = train_step(optimizer, graph, label, net)
    if idx % 100 == 0:
      logging.info('step: %s, loss: %s, acc: %s', idx, scalars['loss'],
                   scalars['accuracy'])
  if save_dir is not None:
    with pathlib.Path(save_dir, 'molhiv.pkl').open('wb') as fp:
      logging.info('Saving model to %s', save_dir)
      pickle.dump(optimizer.target, fp)
  logging.info('Training finished')


def evaluate(data_path, master_csv_path, split_path, save_dir):
  """Evaluation Script."""
  logging.info('Evaluating OGB molviv')
  logging.info('Dataset split: %s', split_path)
  # Initialize the dataset reader.
  reader = data_utils.DataReader(
      data_path=data_path,
      master_csv_path=master_csv_path,
      split_path=split_path,
      batch_size=1)

  with pathlib.Path(save_dir, 'molhiv.pkl').open('rb') as fp:
    params = pickle.load(fp)
  accumulated_loss = 0
  accumulated_accuracy = 0
  idx = 0

  # We jit the computation of our loss, since this is the main computation.
  # Using jax.jit means that we will use a single accelerator. If you want
  # to use more than 1 accelerator, use jax.pmap. More information can be
  # found in the jax documentation.
  net = GraphNetwork(mlp_features=[128, 128], latent_size=128)
  compute_loss_fn = jax.jit(functools.partial(compute_loss, net=net))
  for graph in reader:

    # Jax will re-jit your graphnet every time a new graph shape is encountered.
    # In the limit, this means a new compilation every training step, which
    # will result in *extremely* slow training. To prevent this, pad each
    # batch of graphs to the nearest power of two. Since jax maintains a cache
    # of compiled programs, the compilation cost is amortized.
    graph = pad_graph_to_nearest_power_of_two(graph)

    # Extract the labels and remove from the graph.
    label = graph.globals['label']
    graph = graph._replace(globals={})
    loss, acc = compute_loss_fn(params, graph, label)
    accumulated_accuracy += acc
    accumulated_loss += loss
    idx += 1
    if idx % 100 == 0:
      logging.info('Evaluated %s graphs', idx)
  logging.info('Completed evaluation.')
  loss = accumulated_loss / idx
  accuracy = accumulated_accuracy / idx
  logging.info('Eval loss: %s, accuracy %s', loss, accuracy)
  return loss, accuracy


def main(_):
  if FLAGS.mode == 'train':
    train(FLAGS.data_path, FLAGS.master_csv_path, FLAGS.split_path,
          FLAGS.batch_size, FLAGS.num_training_steps, FLAGS.save_dir)
  elif FLAGS.mode == 'evaluate':
    evaluate(FLAGS.data_path, FLAGS.master_csv_path, FLAGS.split_path,
             FLAGS.save_dir)

if __name__ == '__main__':
  app.run(main)
