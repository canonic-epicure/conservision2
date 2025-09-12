from __future__ import annotations
from typing import List

import torch
import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2

from lib.metric import Metric, Loss, ProbabilitiesMetric


class Epoch():
    epoch: int

    loss: List[float]

    def __init__(self, *args, model, epoch=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = epoch


class Inference():
    name: str       = 'Unknown model'

    model: any

    num_classes: int = 0

    def __init__(self, *args, name='Unknown model', model, num_classes=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.num_classes = num_classes


    def set_input_to_batch(self, input, batch):
        return (input,) + batch[1:]

    def batch_to_input(self, batch):
        return batch[0]


    def set_label_to_batch(self, label, batch):
        return (batch[0], label) + batch[2:]

    def batch_to_label(self, batch):
        return batch[1]


    def batch_to_idx(self, batch):
        return batch[-1]


    def batch_size(self, batch):
        return batch[0].size(0)


    def preprocess_batch_hook(self, batch):
        return batch


    def infer(self, data_loader: DataLoader, desc='Predicting', metrics: List[Metric]=[], T=1):
        for metric in metrics:
            metric.start(len(data_loader.dataset), len(data_loader), self.num_classes)

        self.model.eval()
        with torch.inference_mode():
            for idx, batch in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=desc):
                batch = self.preprocess_batch_hook(batch)

                input = self.batch_to_input(batch).to('cuda')
                output = self.model(input)

                output.logits /= T

                for metric in metrics:
                    metric.update(self.model, input, output, batch)


    def predict_proba(self, data_loader: DataLoader, desc='Predicting', T=1):
        probs = ProbabilitiesMetric()

        self.infer(data_loader, desc, [probs], T)

        return probs


class Training(Inference):
    optimizer: any

    def __init__(self, *args, optimizer, **kwargs):
        super().__init__(*args, **kwargs)

        self.optimizer = optimizer


    def start_epoch(self, epoch):
        self.epoch = epoch


    def train(self, data_loader: DataLoader, desc='Predicting', metrics: List[Metric]=[], T=1):
        loss = None

        for metric in metrics:
            metric.start(len(data_loader.dataset), len(data_loader), self.num_classes)

            if isinstance(metric, Loss):
                if loss is not None:
                    raise ValueError("Only one loss metric can be provided")
                loss = metric

        if loss is None:
            raise ValueError("No loss metric provided")

        self.model.train()
        with torch.train_mode():
            for idx, batch in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=desc):
                batch = self.preprocess_batch_hook(batch)

                input = self.batch_to_input(batch).to('cuda')
                output = self.model(input)

                output.logits /= T

                for metric in metrics:
                    metric.update(self.model, input, output, batch)

                loss.backward()

                self.optimizer.step()


class TrainingWithCutmixMixup(Training):
    def __init__(self, *args, cutmix=None, mixup=None, **kwargs):
        super().__init__(*args, **kwargs)

        transform = []

        cutmix = cutmix if cutmix is not None else v2.CutMix(num_classes=self.num_classes)

        if cutmix:
            transform.append(cutmix)

        mixup = mixup if mixup is not None else v2.MixUp(num_classes=self.num_classes)

        if mixup:
            transform.append(mixup)

        self.transform = v2.RandomChoice(transform)


    def preprocess_batch_hook(self, batch):
        self.set

        return super().preprocess_batch_hook(self.transform(images, labels))
