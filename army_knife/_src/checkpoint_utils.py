"""Heavily based on https://github.com/deepmind/dm-haiku/issues/18"""
import pickle
from pathlib import Path

import numpy as np

import jax

import chex


def save_pytree(tree: chex.ArrayTree, save_path: Path, name: str) -> tuple[Path, Path]:
    assert save_path.is_dir()

    arr_file = save_path / f"{name}.npy"
    tree_file = save_path / f"{name}.pkl"

    with open(arr_file, "wb") as f:
        for x in jax.tree_util.tree_leaves(tree):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, tree)
    with open(tree_file, "wb") as f:
        pickle.dump(tree_struct, f)

    return arr_file, tree_file


def load_pytree(save_path: Path, name: str) -> chex.ArrayTree:
    assert save_path.is_dir()

    with open(save_path / f"{name}.pkl", "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
    with open(save_path / f"{name}.npy", "rb") as f:
        leaves = [np.load(f) for _ in leaves]

    return jax.tree_util.tree_unflatten(treedef, leaves)
