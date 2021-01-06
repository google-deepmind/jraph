Jraph API
=========

.. currentmodule:: jraph

GraphsTuple
-----------

.. autoclass:: GraphsTuple

Batching & Padding Utilities
----------------------------

.. autofunction:: batch

.. autofunction:: unbatch

.. autofunction:: pad_with_graphs

.. autofunction:: get_number_of_padding_with_graphs_graphs

.. autofunction:: get_number_of_padding_with_graphs_nodes

.. autofunction:: get_number_of_padding_with_graphs_edges

.. autofunction:: unpad_with_graphs

.. autofunction:: get_node_padding_mask

.. autofunction:: get_edge_padding_mask

.. autofunction:: get_graph_padding_mask

Segment Utilities
-----------------

.. autofunction:: segment_mean

.. autofunction:: segment_max

.. autofunction:: segment_softmax

.. autofunction:: partition_softmax

Misc Utilities
-----------------

.. autofunction:: concatenated_args

Models
======

.. autofunction:: GraphNetwork

.. autofunction:: InteractionNetwork

.. autofunction:: GraphMapFeatures

.. autofunction:: RelationNetwork

.. autofunction:: DeepSets

.. autofunction:: GraphNetGAT

.. autofunction:: GAT

.. autofunction:: GraphConvolution
