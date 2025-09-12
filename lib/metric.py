from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from lib.training import Training, Inference


class Metric():
    name: str = 'Unknown metric'

    def __init__(self, *args, name, **kwargs):
        super().__init__(*args, **kwargs)

        if name is not None:
            self.name = name

    def start(self, num_samples, num_batches):
        pass

    def update(self, model, input, output, batch):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError


class Loss(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def backward(self, model: Training, input, output, batch):
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    ce: torch.nn.CrossEntropyLoss
    last_value: float

    acc: float
    count : int

    track : List[float]

    def __init__(self, *args, ce=None, **kwargs):
        self.name = 'Cross-entropy'

        super().__init__(*args, **kwargs)

        self.acc = 0
        self.count = 0
        self.track = []

        self.ce = ce if ce is not None else torch.nn.CrossEntropyLoss()
        self.last_value = 0.0

    def start(self, num_samples, num_batches, num_classes):
        self.acc = 0
        self.count = 0
        self.track = []

    def push(self, value, batch_size=1):
        self.acc += value * batch_size
        self.count += batch_size

    def value(self):
        return self.acc / self.count

    def update(self, model: Inference, input, output, batch):
        self.last_value = self.ce(output.logits, model.batch_to_label(batch))
        self.push(self.last_value)
        self.track.append(self.last_value)

    def backward(self, model: Training, input, output, batch):
        self.last_value.backward()


class ProbabilitiesMetric(Metric):
    pos: int
    probs : torch.Tensor
    idx : np.array

    num_samples: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pos = 0

    def start(self, num_samples, num_batches, num_classes):
        self.num_samples = num_samples

        self.probs = torch.zeros((num_samples, num_classes), dtype=torch.float32)
        self.idx = np.empty((num_samples), dtype=np.dtypes.ObjectDType)

    def value(self):
        return self.probs

    def value_as_df(self, columns=None):
        return pd.DataFrame(
            self.probs.detach().to('cpu').numpy(),
            index=self.idx,
            columns=columns,
        )

    def update(self, model, input, output, batch):
        preds = F.softmax(output.logits, dim=1)

        batch_size = model.batch_size(batch)

        if self.pos >= self.num_samples:
            raise OverflowError("Out of bounds")

        self.probs[self.pos:self.pos + batch_size] = preds
        self.idx[self.pos:self.pos + batch_size] = model.batch_to_idx(batch)

        self.pos += batch_size


class AccuracyMetric(Metric):
    pos: int
    probs : torch.Tensor
    idx : np.array

    num_samples: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pos = 0

    def start(self, num_samples, num_batches, num_classes):
        self.num_samples = num_samples

        self.probs = torch.zeros((num_samples, num_classes), dtype=torch.float32)
        self.idx = np.empty((num_samples), dtype=np.dtypes.ObjectDType)

    def value(self):
        return self.probs

    def value_as_df(self, columns=None):
        return pd.DataFrame(
            self.probs.detach().to('cpu').numpy(),
            index=self.idx,
            columns=columns,
        )

    def update(self, model, input, output, batch):
        preds = F.softmax(output.logits, dim=1)

        batch_size = model.batch_size(batch)

        if self.pos >= self.num_samples:
            raise OverflowError("Out of bounds")

        self.probs[self.pos:self.pos + batch_size] = preds
        self.idx[self.pos:self.pos + batch_size] = model.batch_to_idx(batch)

        self.pos += batch_size
