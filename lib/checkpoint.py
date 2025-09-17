import os
import re
from pathlib import Path
from typing import List, Union
import random
import numpy as np
import torch


class CheckpointStorage():
    dir : Path
    pattern : str

    def __init__(self, *args, dir: Path, pattern: str, reg: str, **kwargs):
        if not isinstance(reg, Path): self.dir = Path(dir)

        self.pattern = pattern

        if not isinstance(reg, re.Pattern): reg = re.compile(reg)

        self.reg = reg

        super().__init__(*args, **kwargs)

    def at_epoch(self, idx):
        return self.dir / re.sub(pattern=r'\*', repl=str(idx).rjust(3, '0'), string=self.pattern)

    def touch(self):
        self.dir.mkdir(parents=True, exist_ok=True)

    def latest(self) -> (Path, int):
        if not self.dir.exists():
            return None, None

        def collect():
            for root, dirs, files in os.walk(self.dir):
                for file in files:
                    match = self.reg.match(file)
                    if match:
                        yield file, int(match.group(1))

        epochs = list(collect())

        if len(epochs) == 0:
            return None, None

        epochs.sort(key=lambda res: res[1], reverse=True)

        return self.dir / epochs[0][0], epochs[0][1]


def get_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(rng_state: dict):
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    if torch.cuda.is_available() and rng_state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


def set_seed(s, reproducible=False):
    try: torch.manual_seed(s)
    except NameError: pass

    try: torch.cuda.manual_seed_all(s)
    except NameError: pass

    try: np.random.seed(s%(2**32-1))
    except NameError: pass

    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
