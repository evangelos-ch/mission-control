from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np

from jaxtyping import Array, Float

import chex

Scalar = Float[Union[Array, np.ndarray], ""] | float
Gradients = Float[Union[Array, np.ndarray], "..."]

Image = Float[Union[Array, np.ndarray], "*batch 3 height width"]
Video = Float[Image, "timestep"]


class Logger(metaclass=ABCMeta):
    def __init__(self, log_dir: str | Path):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(exist_ok=True, parents=True)

    @abstractmethod
    def log_metrics(self, metrics: dict, global_step: int):
        raise NotImplementedError

    @abstractmethod
    def log_image(self, name: str, image: Image, global_step: int):
        raise NotImplementedError

    @abstractmethod
    def log_video(self, name: str, video: Video, global_step: int, fps: int = 4):
        raise NotImplementedError

    @abstractmethod
    def log_gradients(self, name: str, gradients: Gradients, global_step: int):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


class WandbLogger(Logger):
    def __init__(self, log_dir: str | Path, run_name: str, project: str, config: dict, job_type: str = "train"):
        try:
            import wandb

            self._wandb = wandb
        except ImportError:
            raise ImportError("Wandb is not installed! Please install it with `pip install army-knife[wandb]`.")

        super().__init__(log_dir)
        self._run = self._wandb.init(project=project, config=config, group=run_name, job_type=job_type)

    def log_metrics(self, metrics: dict[str, Scalar], global_step: int):
        assert self._run is not None
        if "global_step" not in metrics:
            metrics["global_step"] = global_step
        self._run.log(metrics)

    def log_image(self, name: str, image: Image, global_step: int):
        assert self._run is not None
        try:
            chex.assert_rank(image, 3)
            chex.assert_axis_dimension(image, 0, 3)
        except AssertionError:
            raise ValueError("Image must be in CHW format!")
        self._run.log({name: self._wandb.Image(image), global_step: global_step})

    def log_video(self, name: str, video: Video, global_step: int, fps: int = 4):
        assert self._run is not None
        try:
            chex.assert_rank(video, 4)
            chex.assert_axis_dimension(video, 1, 3)
        except AssertionError:
            raise ValueError("Video must be in TCHW format!")
        self._run.log({name: self._wandb.Video(video, fps=fps, format="mp4"), global_step: global_step})

    def log_gradients(self, name: str, gradients: Gradients, global_step: int):
        assert self._run is not None
        self._run.log({name: self._wandb.Histogram(gradients), global_step: global_step})
