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
r"""Higgs Boson Detection Example.

One of the decay-channels of the Higgs Boson is Higgs to two photons.
The two photons must have a combined invariant mass 125 GeV.
In this example we use a relational vector to detect if a Higgs Boson
is present in a set of photons.

There are two situations:
a) Higgs: Two photons with an invariant mass of 125 GeV + an arbitrary number of
   uncorrelated photons.
b) No Higgs: Just an arbitrary number of uncorrelation photons.
"""

import collections
import logging
import random

from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import scipy.stats


Problem = collections.namedtuple("Problem", ("graph", "labels"))


def get_random_rotation_matrix():
  rotation = np.eye(4)
  rotation[1:, 1:] = scipy.stats.ortho_group.rvs(3)
  return rotation


def get_random_boost_matrix():
  eta = np.random.uniform(-1, 1)
  boost = np.eye(4)
  boost[:2, :2] = np.array([[np.cosh(eta), -np.sinh(eta)],
                            [-np.sinh(eta), np.cosh(eta)]])
  rotation = get_random_rotation_matrix()
  return rotation.T @ boost @ rotation


def get_random_higgs_photons():
  higgs = 125.18
  boost = get_random_boost_matrix()
  rotation = get_random_rotation_matrix()
  photon1 = boost @ rotation @ np.array([higgs / 2, higgs / 2, 0, 0])
  photon2 = boost @ rotation @ np.array([higgs / 2, -higgs / 2, 0, 0])
  return photon1, photon2


def get_random_background_photon():
  boost = get_random_boost_matrix()
  rotation = get_random_rotation_matrix()
  energy = np.random.uniform(20, 120)
  return boost @ rotation @ np.array([energy, energy, 0, 0])


def get_higgs_problem(min_n_photons: int, max_n_photons: int) -> Problem:
  """Creates fully connected graph containing the detected photons.

  Args:
    min_n_photons: minimum number of photons in the detector.
    max_n_photons: maximum number of photons in the detector.

  Returns:
    graph, one-hot label whether a higgs was present or not.
  """
  assert min_n_photons >= 2, "Number of photons must be at least 2."
  n_photons = random.randint(min_n_photons, max_n_photons)
  photons = np.stack([get_random_background_photon() for _ in range(n_photons)])

  # Add a higgs
  if random.random() > 0.5:
    label = np.eye(2)[0]
    photons[:2] = np.stack(get_random_higgs_photons())
  else:
    label = np.eye(2)[1]

  # The graph is fully connected.
  senders = np.repeat(np.arange(n_photons), n_photons)
  receivers = np.tile(np.arange(n_photons), n_photons)
  graph = jraph.GraphsTuple(
      n_node=np.asarray([n_photons]),
      n_edge=np.asarray([len(senders)]),
      nodes=photons,
      edges=None,
      globals=None,
      senders=senders,
      receivers=receivers)

  # In order to jit compile our code, we have to pad the nodes and edges of
  # the GraphsTuple to a static shape.
  graph = jraph.pad_with_graphs(graph, max_n_photons + 1,
                                max_n_photons * max_n_photons)

  return Problem(graph=graph, labels=label)


def network_definition(
    graph: jraph.GraphsTuple) -> jraph.ArrayTree:
  """Defines a graph neural network.

  Args:
    graph: Graphstuple the network processes.

  Returns:
    globals.
  """

  @jax.vmap
  @jraph.concatenated_args
  def update_edge_fn(features):
    return hk.nets.MLP([30, 30, 30])(features)

  # The correct solution for the edge update function is the invariant mass
  # of the photon pair.
  # The simple MLP we use here seems to fail to find the correct solution.
  # You can ensure that the example works in principle by replacing the
  # update_edge_fn below with the following analytical solution.
  @jax.vmap
  def unused_update_edge_fn_solution(s, r):
    """Calculates invariant mass of photon pair and compares to Higgs mass."""
    t = (s + r)**2
    return jnp.array(jnp.abs(t[0] - t[1] - t[2] - t[3] - 125.18**2) < 1,
                     dtype=jnp.float32)[None]

  gn = jraph.RelationNetwork(
      update_edge_fn=update_edge_fn,
      update_global_fn=hk.nets.MLP([2]),
      aggregate_edges_for_globals_fn=jraph.segment_sum,
      )
  graph = gn(graph)

  return graph.globals


def train(num_steps: int):
  """Trains a graph neural network on an electronic voting problem."""
  train_dataset = (2, 15)
  test_dataset = (16, 20)
  random.seed(42)

  network = hk.without_apply_rng(hk.transform(network_definition))
  problem = get_higgs_problem(*train_dataset)
  params = network.init(jax.random.PRNGKey(42), problem.graph)

  @jax.jit
  def prediction_loss(params, problem):
    globals_ = network.apply(params, problem.graph)
    # We interpret the globals as logits for the detection.
    # Only the first graph is real, the second graph is for padding.
    log_prob = jax.nn.log_softmax(globals_[0]) * problem.labels
    return -jnp.sum(log_prob)

  @jax.jit
  def accuracy_loss(params, problem):
    globals_ = network.apply(params, problem.graph)
    # We interpret the globals as logits for the detection.
    # Only the first graph is real, the second graph is for padding.
    equal = jnp.argmax(globals_[0]) == jnp.argmax(problem.labels)
    return equal.astype(np.int32)

  opt_init, opt_update = optax.adam(2e-4)
  opt_state = opt_init(params)

  @jax.jit
  def update(params, opt_state, problem):
    g = jax.grad(prediction_loss)(params, problem)
    updates, opt_state = opt_update(g, opt_state)
    return optax.apply_updates(params, updates), opt_state

  for step in range(num_steps):
    problem = get_higgs_problem(*train_dataset)
    params, opt_state = update(params, opt_state, problem)
    if step % 1000 == 0:
      train_loss = jnp.mean(
          jnp.asarray([
              accuracy_loss(params, get_higgs_problem(*train_dataset))
              for _ in range(100)
          ])).item()
      test_loss = jnp.mean(
          jnp.asarray([
              accuracy_loss(params, get_higgs_problem(*test_dataset))
              for _ in range(100)
          ])).item()
      logging.info("step %r loss train %r test %r", step, train_loss, test_loss)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  train(num_steps=10000)


if __name__ == "__main__":
  app.run(main)
