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
"""Jraph."""


from jraph._src.graph import GraphsTuple
from jraph._src.models import AggregateEdgesToGlobalsFn
from jraph._src.models import AggregateEdgesToNodesFn
from jraph._src.models import AggregateNodesToGlobalsFn
from jraph._src.models import AttentionLogitFn
from jraph._src.models import AttentionReduceFn
from jraph._src.models import DeepSets
from jraph._src.models import EmbedEdgeFn
from jraph._src.models import EmbedGlobalFn
from jraph._src.models import EmbedNodeFn
from jraph._src.models import GAT
from jraph._src.models import GATAttentionLogitFn
from jraph._src.models import GATAttentionQueryFn
from jraph._src.models import GATNodeUpdateFn
from jraph._src.models import GNUpdateEdgeFn
from jraph._src.models import GNUpdateGlobalFn
from jraph._src.models import GNUpdateNodeFn
from jraph._src.models import GraphConvolution
from jraph._src.models import GraphMapFeatures
from jraph._src.models import GraphNetGAT
from jraph._src.models import GraphNetwork
from jraph._src.models import InteractionNetwork
from jraph._src.models import InteractionUpdateEdgeFn
from jraph._src.models import InteractionUpdateNodeFn
from jraph._src.models import NodeFeatures
from jraph._src.models import RelationNetwork
from jraph._src.utils import ArrayTree
from jraph._src.utils import batch
from jraph._src.utils import batch_np
from jraph._src.utils import concatenated_args
from jraph._src.utils import dynamically_batch
from jraph._src.utils import get_edge_padding_mask
from jraph._src.utils import get_fully_connected_graph
from jraph._src.utils import get_graph_padding_mask
from jraph._src.utils import get_node_padding_mask
from jraph._src.utils import get_number_of_padding_with_graphs_edges
from jraph._src.utils import get_number_of_padding_with_graphs_graphs
from jraph._src.utils import get_number_of_padding_with_graphs_nodes
from jraph._src.utils import pad_with_graphs
from jraph._src.utils import partition_softmax
from jraph._src.utils import segment_max
from jraph._src.utils import segment_max_or_constant
from jraph._src.utils import segment_mean
from jraph._src.utils import segment_min
from jraph._src.utils import segment_min_or_constant
from jraph._src.utils import segment_normalize
from jraph._src.utils import segment_softmax
from jraph._src.utils import segment_sum
from jraph._src.utils import segment_variance
from jraph._src.utils import sparse_matrix_to_graphs_tuple
from jraph._src.utils import unbatch
from jraph._src.utils import unbatch_np
from jraph._src.utils import unpad_with_graphs
from jraph._src.utils import with_zero_out_padding_outputs
from jraph._src.utils import zero_out_padding


__version__ = "0.0.6.dev0"

__all__ = ("ArrayTree", "DeepSets", "GraphConvolution", "GraphMapFeatures",
           "InteractionNetwork", "RelationNetwork", "GraphNetGAT", "GAT",
           "GraphsTuple", "GraphNetwork", "NodeFeatures",
           "AggregateEdgesToNodesFn", "AggregateNodesToGlobalsFn",
           "AggregateEdgesToGlobalsFn", "AttentionLogitFn", "AttentionReduceFn",
           "GNUpdateEdgeFn", "GNUpdateNodeFn", "GNUpdateGlobalFn",
           "InteractionUpdateNodeFn", "InteractionUpdateEdgeFn", "EmbedEdgeFn",
           "EmbedNodeFn", "EmbedGlobalFn", "GATAttentionQueryFn",
           "GATAttentionLogitFn", "GATNodeUpdateFn", "batch", "batch_np",
           "unbatch", "unbatch_np", "pad_with_graphs",
           "get_number_of_padding_with_graphs_graphs",
           "get_number_of_padding_with_graphs_nodes",
           "get_number_of_padding_with_graphs_edges", "unpad_with_graphs",
           "get_node_padding_mask", "get_edge_padding_mask",
           "get_graph_padding_mask", "segment_max", "segment_max_or_constant",
           "segment_min_or_constant", "segment_softmax", "segment_sum",
           "partition_softmax", "concatenated_args",
           "get_fully_connected_graph", "dynamically_batch",
           "with_zero_out_padding_outputs", "zero_out_padding",
           "sparse_matrix_to_graphs_tuple")

#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the Jraph public API.    /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
try:
  del _src  # pylint: disable=undefined-variable
except NameError:
  pass
