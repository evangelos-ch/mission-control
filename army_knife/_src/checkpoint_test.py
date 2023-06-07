"""Heavily based on https://github.com/kevinzakka/torchkit"""
import jax
import jax.numpy as jnp
import optax

import chex
import haiku as hk

import pytest

from .checkpoint import Checkpoint, CheckpointManager


@pytest.fixture
def init_model_and_optimizer():
    init_fn, apply_fn = hk.without_apply_rng(hk.transform(lambda x: hk.Linear(1)(x)))
    key = jax.random.PRNGKey(42)
    params = init_fn(rng=key, x=jnp.arange(300, dtype=jnp.float32))
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)
    return apply_fn, key, params, opt, opt_state


def test_checkpoint_save_restore(tmp_path, init_model_and_optimizer):
    _, key, params, _, opt_state = init_model_and_optimizer
    checkpoint = Checkpoint(model_params=params, adam_state=opt_state, rng=key)
    arr_file, _ = checkpoint.save(tmp_path, "test")
    assert arr_file.parent == tmp_path
    ckpt = Checkpoint.load(arr_file.parent, arr_file.stem)
    assert ckpt
    chex.assert_tree_all_close(ckpt.model_params, params)


def test_checkpoint_manager(tmp_path, init_model_and_optimizer):
    _, key, params, _, opt_state = init_model_and_optimizer
    ckpt_dir = tmp_path / "ckpts"
    checkpoint_manager = CheckpointManager(
        ckpt_dir,
        max_to_keep=5,
    )
    tree, global_step = checkpoint_manager.restore_or_initialize()
    assert global_step == 0
    assert tree is None
    for i in range(10):
        checkpoint_manager.save(i, model_params=params, adam_state=opt_state, rng=key)
    available_ckpts = CheckpointManager.list_checkpoints(ckpt_dir)
    assert len(available_ckpts) == 5
    ckpts = [int(d.stem) for d in available_ckpts]
    expected = list(range(5, 10))
    assert all([a == b for a, b in zip(ckpts, expected)])
    ckpt, global_step = checkpoint_manager.restore_or_initialize()
    assert global_step == 9
    assert int(checkpoint_manager.latest_checkpoint.stem) == 9
