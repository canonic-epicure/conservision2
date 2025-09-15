import os
import re
from pathlib import Path
from typing import List, Union


class CheckpointStorage():
    dir : Path

    def __init__(self, *args, dir: Path, **kwargs):
        self.dir = dir
        super().__init__(*args, **kwargs)


    def touch(self):
        self.dir.mkdir(parents=True, exist_ok=True)

    def latest(self, reg: Union[re.Pattern, str]) -> (Path, int):
        if not self.dir.exists():
            return None, None

        if not isinstance(reg, re.Pattern):
            reg = re.compile(reg)

        def collect():
            for root, dirs, files in os.walk(self.dir):
                for file in files:
                    match = reg.match(file)
                    if match:
                        yield file, int(match.group(1))

        epochs = list(collect())

        if len(epochs) == 0:
            return None, None

        epochs.sort(key=lambda res: res[1], reverse=True)

        return self.dir / epochs[0][0], epochs[0][1]
