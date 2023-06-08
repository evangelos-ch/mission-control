"""Heavily based on https://github.com/kevinzakka/torchkit"""
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np

import jax
from jaxtyping import Array, Float, Num

import chex

from .utils import unique_id

try:
    import wandb

    WANDB_INSTALLED = True
except ImportError:  # pragma: no cover
    WANDB_INSTALLED = False

try:
    import tensorboardX

    TENSORBOARD_INSTALLED = True
except ImportError:
    TENSORBOARD_INSTALLED = False


Scalar = Num[Union[Array, np.ndarray], ""] | float | int
Gradients = Float[Union[Array, np.ndarray], "..."]

Image = Num[Union[Array, np.ndarray], "height width 3"]
Video = Num[Image, "timestep"]


class Logger(metaclass=ABCMeta):
    def __init__(self, log_dir: str | Path):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(exist_ok=True, parents=True)

    @abstractmethod
    def log_metrics(self, metrics: dict, global_step: int):
        raise NotImplementedError

    @abstractmethod
    def log_image(self, image: Image, name: str, global_step: int):
        raise NotImplementedError

    @abstractmethod
    def log_video(self, video: Video, name: str, global_step: int, fps: int = 4):
        raise NotImplementedError

    @abstractmethod
    def log_gradients(self, gradients: Gradients, name: str, global_step: int):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


class WandbLogger(Logger):
    def __init__(
        self,
        log_dir: str | Path,
        run_name: str,
        project: str,
        config: dict,
        job_type: str = "train",
        run_id: str = unique_id(),
    ):
        if not WANDB_INSTALLED:
            raise ImportError("Wandb is not installed! Please install it with `pip install army-knife[wandb]`.")

        super().__init__(log_dir)
        self._run = wandb.init(
            project=project,
            config=config,
            name=run_name,
            id=run_id,
            job_type=job_type,
            resume=True,
            dir=str(self._log_dir),
        )

    def log_metrics(self, metrics: dict[str, Scalar], global_step: int):
        assert self._run is not None
        for key, value in metrics.items():
            try:
                chex.assert_rank(value, 0)
            except AssertionError:
                if len(value.shape) == 1 and value.shape[0] == 1:
                    metrics[key] = value[0]
                else:
                    raise ValueError("Metrics must be scalars!")
        if "global_step" not in metrics:
            metrics["global_step"] = global_step
        self._run.log(metrics)

    def log_image(self, image: Image, name: str, global_step: int):
        assert self._run is not None
        try:
            chex.assert_rank(image, 3)
            chex.assert_axis_dimension(image, -1, 3)
        except AssertionError:  # pragma: no cover
            raise TypeError("Image must be in HWC format!")
        self._run.log({name: wandb.Image(jax.device_get(image)), "global_step": global_step})

    def log_video(self, video: Video, name: str, global_step: int, fps: int = 4):
        assert self._run is not None
        try:
            chex.assert_rank(video, 4)
            chex.assert_axis_dimension(video, -1, 3)
        except AssertionError:  # pragma: no cover
            raise TypeError("Video must be in THWC format!")
        self._run.log({name: wandb.Video(jax.device_get(video), fps=fps, format="mp4"), "global_step": global_step})

    def log_gradients(self, gradients: Gradients, name: str, global_step: int):
        assert self._run is not None
        self._run.log({name: wandb.Histogram(jax.device_get(gradients)), "global_step": global_step})

    def close(self):
        if self._run:
            self._run.finish()


class TensorboardLogger(Logger):
    def __init__(self, log_dir: str | Path):
        super().__init__(log_dir)
        if not TENSORBOARD_INSTALLED:
            raise ImportError(
                "tensorboardX is not installed! Please install it with `pip install army-knife[tensorboard]`."
            )

        self._writer = tensorboardX.SummaryWriter(str(log_dir))

    def log_metrics(self, metrics: dict[str, Scalar], global_step: int):
        for key, value in metrics.items():
            try:
                chex.assert_rank(value, 0)
            except AssertionError:
                if len(value.shape) == 1 and value.shape[0] == 1:
                    metrics[key] = value[0]
                else:
                    raise ValueError("Metrics must be scalars!")
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, global_step)

    def log_image(self, image: Image, name: str, global_step: int):
        try:
            chex.assert_rank(image, 3)
            chex.assert_axis_dimension(image, -1, 3)
        except AssertionError:  # pragma: no cover
            raise TypeError("Image must be in HWC format!")
        image = jax.device_get(image)
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW because Tensorboard
        self._writer.add_image(name, image, global_step)

    def log_video(self, video: Video, name: str, global_step: int, fps: int = 4):
        try:
            chex.assert_rank(video, 4)
            chex.assert_axis_dimension(video, -1, 3)
        except AssertionError:  # pragma: no cover
            raise TypeError("Video must be in THWC format!")
        video = jax.device_get(video)
        video = np.transpose(video, (0, 3, 1, 2))  # THWC -> TCHW because Tensorboard
        video = video[None, ...]  # Add batch dimension
        self._writer.add_video(name, video, global_step, fps=fps)

    def log_gradients(self, gradients: Gradients, name: str, global_step: int):
        self._writer.add_histogram(name, jax.device_get(gradients), global_step)

    def close(self):
        self._writer.close()
