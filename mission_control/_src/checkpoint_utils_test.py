import jax
import jax.numpy as jnp

import chex
import haiku as hk
import optax  # type: ignore

from .checkpoint_utils import load_pytree, save_pytree


def test_save_load_pytree(tmp_path):
    rng_key = jax.random.PRNGKey(42)
    init_fn, _ = hk.without_apply_rng(hk.transform(lambda x: hk.Linear(1)(x)))
    rng_key, init_key = jax.random.split(rng_key)
    params = init_fn(rng=init_key, x=jnp.arange(300, dtype=jnp.float32))
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    tree = {"model": params, "opt": opt_state, "rng": rng_key, "step": 5, "name": "test_thing"}
    save_pytree(tree, tmp_path, "test_thing_5")
    assert (tmp_path / "test_thing_5.npy").exists()
    assert (tmp_path / "test_thing_5.pkl").exists()

    tree2 = load_pytree(tmp_path, "test_thing_5")
    chex.assert_tree_all_close(tree2["model"], params)
    chex.assert_tree_all_close(tree2["opt"], opt_state)
    chex.assert_tree_all_close(tree2["rng"], rng_key)
    assert tree2["step"] == 5
    assert tree2["name"] == "test_thing"
