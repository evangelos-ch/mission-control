# mission-control

![CI Status](https://github.com/evangelos-ch/mission-control/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/evangelos-ch/army-knife/branch/main/graph/badge.svg?token=zUVZKBavVI)](https://codecov.io/gh/evangelos-ch/army-knife)

A set of utilities for retaining sanity while managing and monitoring ML experiments that utilise models written in JAX.

This is for my own personal use as I'm trying to build a tech stack that allows me to rapidly conduct RL experiments for my doctorate. I plan to add more stuff to this as I need to.

**_HEAVILY_** inspired by & based on the code of [torchkit](https://github.com/kevinzakka/torchkit). My main goal was to essentially replicate it but make it compatible with JAX patterns.

Currently has the following utilities:

-   `mission_control.checkpoint`: Provides a `CheckpointManager` class that can be used to save & load `Checkpoint`s of arbitrary PyTrees, e.g. haiku Params, optax OptStates, JAX arrays, numPy arrays and any other PyTree whose leaves can be serialized with `np.save`. The interface just requires that you provide PyTrees as kwargs.
    > Note that the solution to checkpointing I went with uses `pickle` to save the `treedef` and is thus far from ideal. However, it was simple enough and it will do for my usecase, ~~and it also seems to be used by [other practicioners](https://github.com/deepmind/dm-haiku/issues/18)~~. For actual "prod" usecases [Orbax](https://github.com/google/orbax) is superior since it actually serializes things to JSON.
-   `mission_control.loggers`: Provides a `Logger` interface for logging common training artifacts such as metrics, images and videos. Currently supports logging to Weights & Biases and Tensorboard with `WandbLogger` and `TensorboardLogger`.

## Installation

```bash
pip install "git+https://github.com/evangelos-ch/mission-control.git"
```

If you want to use the loggers, you need to install the required extras (either `wandb` or `tensorboard`). For example:

```bash
pip install "mission-control[wandb] @ git+https://github.com/evangelos-ch/mission-control.git"
```
