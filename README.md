# Jraph - A library for graph neural networks in jax.

Jraph (pronounced giraffe) is a lightweight library for working with graph
neural networks in jax. It provides a data structure for graphs, a set of
utilites for working with graphs, and a 'zoo' of forkable graph neural network
models.

## Getting Started

The best place to start are the examples: `ogb_examples` provides an end to
end example of training a `GraphNet` on `molhiv` Open Graph Benchmark dataset.
The rest of the examples are short scripts demonstrating how to use various
models from our model zoo.

## Overview

Jraph is designed to provide utilities for
working with graphs in jax, but doesn't prescribe a way to write or develop
graph neural networks.

*   `graph.py` provides a lightweight data structure, `GraphsTuple`, for working
    with graphs.
*   `utils.py` provides utilies for working with `GraphsTuples` in jax.
    *   Utilities for batching datasets of `GraphsTuples`.
    *   Utilities to support jit compilation of variable shaped graphs via
        padding and masking.
    *   Utilities for defining losses on partitions of inputs.
*   `models.py` provides examples of different types of graph neural network
    message passing. These are designed to be lightweight, easy to fork and
    adapt. They do not manage parameters for you - for that, consider using
    `haiku` or `flax`. See the examples for more details.
