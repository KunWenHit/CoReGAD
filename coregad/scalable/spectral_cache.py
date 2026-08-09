from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MemmapSpec:
    path: Path
    shape: tuple[int, ...]
    dtype: str

    @property
    def bytes(self) -> int:
        return int(np.prod(self.shape)) * int(np.dtype(self.dtype).itemsize)

    def open(self, mode: str = "r") -> np.memmap:
        return np.memmap(
            self.path, mode=mode, dtype=np.dtype(self.dtype), shape=self.shape
        )


def create_memmap(
    path: str | Path, shape: tuple[int, ...], dtype: str
) -> tuple[MemmapSpec, np.memmap]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    spec = MemmapSpec(target, tuple(int(value) for value in shape), str(np.dtype(dtype)))
    return spec, spec.open("w+")
