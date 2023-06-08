"""Heavily based on https://github.com/kevinzakka/torchkit"""
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr

import haiku as hk
import optax

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from . import loggers


@pytest.fixture
def init_model_and_optimizer():
    init_fn, apply_fn = hk.without_apply_rng(hk.transform(lambda x: hk.Linear(1)(x)))
    key = jr.PRNGKey(42)
    params = init_fn(rng=key, x=jnp.arange(300, dtype=jnp.float32))
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)
    return apply_fn, key, params, opt, opt_state


@pytest.fixture(params=[loggers.WandbLogger])
def logger(tmp_path, request):
    path = Path(tmp_path) / "logs"
    logger_cls = request.param
    if logger_cls == loggers.WandbLogger:
        logger = loggers.WandbLogger(path, run_name="name", project="project", config={"hparams": 0})
        yield logger
    else:
        raise NotImplementedError
    logger.close()


class TestLogger:
    @pytest.mark.parametrize(
        "loss", [jnp.array([5.0]), jr.normal(key=jr.PRNGKey(42), shape=[2, 2]).mean(), 5.0, np.array([5.0])]
    )
    def test_log_metrics(self, logger: loggers.Logger, loss):
        logger.log_metrics({"loss": loss}, global_step=0)

    def test_log_invalid_metrics(self, logger):
        scalar = jnp.array([5.0, 5.0])
        with pytest.raises(ValueError):
            logger.log_metrics({"loss": scalar}, global_step=0)

    @pytest.mark.parametrize(
        "image",
        [
            np.random.randint(0, 256, size=(224, 224, 3)),
            jr.randint(jr.PRNGKey(42), (224, 224, 3), 0, 256),
        ],
    )
    def test_log_image(self, logger: loggers.Logger, image):
        logger.log_image(image, global_step=0, name="image")

    @pytest.mark.parametrize(
        "image",
        [
            np.random.randint(0, 256, size=(3, 224, 224)),
            np.random.randint(0, 256, size=(2, 224, 224, 3)),
            jr.randint(jr.PRNGKey(42), (3, 224, 224), 0, 256),
            jr.randint(jr.PRNGKey(42), (2, 224, 224, 3), 0, 256),
        ],
    )
    def test_log_image_wrong_format(self, logger: loggers.Logger, image):
        with pytest.raises(BeartypeCallHintParamViolation):
            logger.log_image(image, global_step=0, name="image")

    @pytest.mark.parametrize(
        "video",
        [
            np.random.randint(0, 256, size=(5, 224, 224, 3)),
            jr.randint(jr.PRNGKey(42), (5, 224, 224, 3), 0, 256),
        ],
    )
    def test_log_video(self, logger: loggers.Logger, video):
        logger.log_video(video, global_step=0, name="video")

    def test_log_video_wrongdim(self, logger: loggers.Logger):
        image = np.random.randint(0, 256, (224, 224, 3))
        with pytest.raises(BeartypeCallHintParamViolation):
            logger.log_video(image, global_step=0, name="video")

    def test_log_gradients(self, logger: loggers.Logger, init_model_and_optimizer):
        apply_fn, key, params, opt, opt_state = init_model_and_optimizer
        key, subkey = jr.split(key)
        x = jr.normal(key=subkey, shape=(300,))
        y = jr.normal(key=key, shape=(300,))

        def _loss_fn(_x, _p, _y):
            return jnp.mean(jnp.square(apply_fn(x=_x, params=_p) - _y))

        grad = jax.grad(_loss_fn)(x, params, y)
        logger.log_gradients(grad, global_step=0, name="model_grad")


def test_wandb_not_installed(tmp_path):
    path = Path(tmp_path) / "logs"
    loggers.WANDB_INSTALLED = False

    with pytest.raises(ImportError):
        loggers.WandbLogger(path, run_name="name", project="project", config={"hparams": 0})

    loggers.WANDB_INSTALLED = True
