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
"""Data loading utils for the Open Graph Benchmark (OGB) Mol-Hiv."""

import pathlib
import jraph
import numpy as np
import pandas as pd


class DataReader:
  """Data Reader for Open Graph Benchmark datasets."""

  def __init__(self,
               data_path,
               master_csv_path,
               split_path,
               batch_size=1,
               dynamically_batch=False):
    """Initializes the data reader by loading in data."""
    with pathlib.Path(master_csv_path).open("rt") as fp:
      self._dataset_info = pd.read_csv(fp, index_col=0)["ogbg-molhiv"]
    self._data_path = pathlib.Path(data_path)
    # Load edge information, and transpose into (senders, receivers).
    with pathlib.Path(data_path, "edge.csv.gz").open("rb") as fp:
      sender_receivers = pd.read_csv(
          fp, compression="gzip", header=None).values.T.astype(np.int64)
      self._senders = sender_receivers[0]
      self._receivers = sender_receivers[1]
    # Load n_node and n_edge
    with pathlib.Path(data_path, "num-node-list.csv.gz").open("rb") as fp:
      self._n_node = pd.read_csv(fp, compression="gzip", header=None)
      self._n_node = self._n_node.astype(np.int64)[0].tolist()
    with pathlib.Path(data_path, "num-edge-list.csv.gz").open("rb") as fp:
      self._n_edge = pd.read_csv(fp, compression="gzip", header=None)
      self._n_edge = self._n_edge.astype(np.int64)[0].tolist()
    # Load node features
    with pathlib.Path(data_path, "node-feat.csv.gz").open("rb") as fp:
      self._nodes = pd.read_csv(
          fp, compression="gzip", header=None).astype(np.float32).values
    with pathlib.Path(data_path, "edge-feat.csv.gz").open("rb") as fp:
      self._edges = pd.read_csv(
          fp, compression="gzip", header=None).astype(np.float32).values
    with pathlib.Path(data_path, "graph-label.csv.gz").open("rb") as fp:
      self._labels = pd.read_csv(
          fp, compression="gzip", header=None).values

    with pathlib.Path(split_path).open("rb") as fp:
      self._split_idx = pd.read_csv(
          fp, compression="gzip", header=None).values.T[0]

    self._repeat = False
    self._batch_size = batch_size
    self._generator = self._make_generator()
    self._max_nodes = int(np.max(self._n_node))
    self._max_edges = int(np.max(self._n_edge))

    if dynamically_batch:
      self._generator = jraph.dynamically_batch(
          self._generator,
          # Plus one for the extra padding node.
          n_node=self._batch_size * (self._max_nodes) + 1,
          # Times two because we want backwards edges.
          n_edge=self._batch_size * (self._max_edges) * 2,
          n_graph=self._batch_size + 1)

    # If n_node = [1,2,3], we create accumulated n_node [0,1,3,6] for indexing.
    self._accumulated_n_nodes = np.concatenate((np.array([0]),
                                                np.cumsum(self._n_node)))
    # Same for n_edge
    self._accumulated_n_edges = np.concatenate((np.array([0]),
                                                np.cumsum(self._n_edge)))
    self._dynamically_batch = dynamically_batch

  @property
  def total_num_graphs(self):
    return len(self._n_node)

  def repeat(self):
    self._repeat = True

  def __iter__(self):
    return self

  def __next__(self):
    graphs = []
    if self._dynamically_batch:
      # If we are using pmap we need each batch to have the same size in both
      # number of nodes and number of edges. So we use dynamically batch which
      # guarantees this.
      return next(self._generator)
    else:
      for _ in range(self._batch_size):
        graph = next(self._generator)
        graphs.append(graph)
      return jraph.batch(graphs)

  def get_graph_by_idx(self, idx):
    """Gets a graph by an integer index."""
    # Gather the graph information
    label = self._labels[idx]
    n_node = self._n_node[idx]
    n_edge = self._n_edge[idx]
    node_slice = slice(
        self._accumulated_n_nodes[idx], self._accumulated_n_nodes[idx+1])
    edge_slice = slice(
        self._accumulated_n_edges[idx], self._accumulated_n_edges[idx+1])
    nodes = self._nodes[node_slice]
    edges = self._edges[edge_slice]
    senders = self._senders[edge_slice]
    receivers = self._receivers[edge_slice]
    # Molecular graphs are bi directional, but the serialization only
    # stores one way so we add the missing edges.
    return jraph.GraphsTuple(
        nodes=nodes,
        edges=np.concatenate([edges, edges]),
        n_node=np.array([n_node]),
        n_edge=np.array([n_edge*2]),
        senders=np.concatenate([senders, receivers]),
        receivers=np.concatenate([receivers, senders]),
        globals={"label": label})

  def _make_generator(self):
    """Makes a single example generator of the loaded OGB data."""
    idx = 0
    while True:
      # If not repeating, exit when we've cycled through all the graphs.
      # Only return graphs within the split.
      if not self._repeat:
        if idx == self.total_num_graphs:
          return
      else:
        # This will reset the index to 0 if we are at the end of the dataset.
        idx = idx % self.total_num_graphs
      if idx not in self._split_idx:
        idx += 1
        continue
      graph = self.get_graph_by_idx(idx)
      idx += 1
      yield graph
