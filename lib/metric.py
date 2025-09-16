from __future__ import annotations

from typing import List
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

if TYPE_CHECKING:
    from lib.training import Training, Inference


class Metric():
    id: str
    name: str

    def __init__(self, *args, name='Unknown metric', **kwargs):
        super().__init__(*args, **kwargs)

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
        self.id = 'cross_entropy'
        super().__init__(*args, **kwargs)
        self.name = 'Cross-entropy'

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
        self.last_value = self.ce(output.logits, batch['labels'].to('cuda'))
        self.push(self.last_value)
        self.track.append(self.last_value)

    def backward(self, model: Training, input, output, batch):
        self.last_value.backward()


class ProbabilitiesMetric(Metric):
    pos: int
    probs : torch.Tensor
    ids : np.array

    num_samples: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pos = 0

    def start(self, num_samples, num_batches, num_classes):
        self.num_samples = num_samples

        self.pos = 0
        self.probs = torch.zeros((num_samples, num_classes), dtype=torch.float32)
        self.ids = np.empty((num_samples), dtype=np.dtypes.ObjectDType)

    def value(self):
        return self.probs

    def value_as_df(self, columns=None):
        return pd.DataFrame(
            self.probs.detach().to('cpu').numpy(),
            index=self.ids,
            columns=columns,
        )

    def update(self, model, input, output, batch):
        preds = F.softmax(output.logits, dim=1)

        batch_size = batch['inputs'].size(0)

        if self.pos >= self.num_samples:
            raise OverflowError("Out of bounds")

        self.probs[self.pos:self.pos + batch_size] = preds
        self.ids[self.pos:self.pos + batch_size] = batch['ids']

        self.pos += batch_size


class AccuracyMetric(Metric):
    pos: int
    pred_cls_idx : torch.Tensor
    true_cls_idx: torch.Tensor
    accuracy: torch.Tensor
    ids : np.array

    num_samples: int

    def __init__(self, *args, **kwargs):
        self.id = 'accuracy'

        super().__init__(*args, **kwargs)

        self.name = 'Accuracy'

        self.pos = 0

    def start(self, num_samples, num_batches, num_classes):
        self.num_samples = num_samples

        self.pos = 0
        self.pred_cls_idx = torch.zeros((num_samples), dtype=torch.int8)
        self.true_cls_idx = torch.zeros((num_samples), dtype=torch.int8)
        self.accuracy = torch.zeros((num_samples), dtype=torch.int8)
        self.ids = np.empty((num_samples), dtype=np.dtypes.ObjectDType)

    def value(self):
        return self.accuracy.float().mean()

    def value_as_df(self, columns=None):
        return pd.DataFrame(
            [
                self.pred_cls_idx.detach().to('cpu').numpy(),
                self.true_cls_idx.detach().to('cpu').numpy(),
                self.accuracy.detach().to('cpu').numpy(),
            ],
            index=self.ids,
            columns=columns,
        )

    def update(self, model, input, output, batch):
        preds = F.softmax(output.logits, dim=1)

        pred_cls_idx = preds.argmax(dim=1).to('cpu')

        batch_size = batch['inputs'].size(0)

        if self.pos >= self.num_samples:
            raise OverflowError("Out of bounds")

        self.pred_cls_idx[self.pos:self.pos + batch_size] = pred_cls_idx
        self.true_cls_idx[self.pos:self.pos + batch_size] = batch['labels']
        self.accuracy[self.pos:self.pos + batch_size] = pred_cls_idx == batch['labels']
        self.ids[self.pos:self.pos + batch_size] = batch['ids']

        self.pos += batch_size
