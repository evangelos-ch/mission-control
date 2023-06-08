"""Heavily based on https://github.com/kevinzakka/torchkit"""
import logging
import signal
import tempfile
from pathlib import Path

from jaxtyping import PyTree

from .checkpoint_utils import load_pytree, save_pytree
from .utils import unique_id


class Checkpoint:
    def __init__(self, **kwargs: dict[str, PyTree]):
        for k, v in sorted(kwargs.items()):
            setattr(self, k, v)

    def save(self, save_path: str | Path, name: str) -> tuple[Path, Path]:
        assert Path(save_path).is_dir()

        # Ignore ctrl+c while saving.
        try:
            orig_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, lambda _sig, _frame: None)
        except ValueError:
            # Signal throws a ValueError if we're not in the main thread.
            orig_handler = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save to a temporary directory first, then move
            arr_file, tree_file = save_pytree(self.__dict__, Path(tmp_dir), f"{name}-{unique_id()}")
            arr_file = arr_file.rename(save_path / f"{name}.npy")
            tree_file = tree_file.rename(save_path / f"{name}.pkl")

        # Restore SIGINT handler.
        if orig_handler is not None:
            signal.signal(signal.SIGINT, orig_handler)

        return arr_file, tree_file

    @classmethod
    def load(cls, save_path: str | Path, name: str) -> "Checkpoint":
        tree = load_pytree(Path(save_path), name)
        return cls(**tree)


class CheckpointManager:
    def __init__(self, directory: str | Path, max_to_keep: int = 10):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep

    def save(self, global_step: int, **kwargs: dict[str, PyTree]):
        Checkpoint(global_step=global_step, **kwargs).save(self.directory, str(global_step))
        self._trim_checkpoints()

    def restore_or_initialize(self) -> tuple[Checkpoint | None, int]:
        ckpts = CheckpointManager.list_checkpoints(self.directory)
        if not ckpts:
            return None, 0
        last_ckpt = ckpts[-1]
        ckpt = self._load(last_ckpt.stem)
        if not ckpt:
            logging.info("Could not restore latest checkpoint file.")
            return None, 0
        return ckpt, int(last_ckpt.stem)

    def _load(self, name: str) -> Checkpoint | None:
        return Checkpoint.load(self.directory, name)

    def _trim_checkpoints(self):
        arrs = CheckpointManager.list_checkpoints(self.directory, arrays=True, treedefs=False)[::-1]
        trees = CheckpointManager.list_checkpoints(self.directory, arrays=False, treedefs=True)[::-1]
        checkpoints = list(zip(arrs, trees))
        # Remove until `max_to_keep` remain.
        while len(checkpoints) - self.max_to_keep > 0:
            arr, tree = checkpoints.pop()
            arr.unlink()
            tree.unlink()

    def load_latest_checkpoint(self) -> tuple[Checkpoint, int]:
        return self._load(self.latest_checkpoint.stem), self.latest_checkpoint.stem

    def load_checkpoint_at(self, global_step: int) -> tuple[Checkpoint, int]:
        if str(global_step) not in [x.stem for x in CheckpointManager.list_checkpoints(self.directory)]:
            raise ValueError(f"No checkpoint found at step {global_step}.")
        return self._load(str(global_step)), global_step

    @property
    def latest_checkpoint(self) -> Path | None:
        checkpoints = self.list_checkpoints(self.directory)
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {self.directory}.")
        return checkpoints[-1]

    @staticmethod
    def list_checkpoints(directory: str | Path, arrays: bool = True, treedefs: bool = False) -> list[Path] | None:
        files = []
        directory = Path(directory)
        if arrays:
            files += directory.glob("*.npy")
        if treedefs:
            files += directory.glob("*.pkl")
        return sorted(files, key=lambda x: int(x.stem))
