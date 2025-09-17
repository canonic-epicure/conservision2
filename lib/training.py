from __future__ import annotations

from typing import List

import torch
import torchvision.transforms.v2 as v2
import tqdm
from torch.utils.data import DataLoader

from lib.metric import Metric, Loss, ProbabilitiesMetric

#-----------------------------------------------------------------------------------------------------------------------
class Inference():
    name: str

    model: any

    num_classes: int = 0

    def __init__(self, *args, name='Unknown model', model, num_classes=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.model = model
        self.num_classes = num_classes


    def preprocess_batch_hook(self, batch):
        return batch


    def infer(self, data_loader: DataLoader, desc=None, metrics: List[Metric]=[], T=1):
        for metric in metrics:
            metric.start(len(data_loader.dataset), len(data_loader), self.num_classes)

        self.model.eval()
        with torch.inference_mode():
            for idx, batch in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=desc if desc is not None else f'Inference { self.name }'):
                batch = self.preprocess_batch_hook(batch)

                input = batch['inputs'].to('cuda')
                output = self.model(input)

                output.logits /= T

                for metric in metrics:
                    metric.update(self.model, input, output, batch)


    def predict_proba(self, data_loader: DataLoader, metrics: List[Metric]=[], desc='Predicting', T=1):
        probs = ProbabilitiesMetric()

        self.infer(data_loader, desc, [probs].extend(metrics), T)

        return probs


#-----------------------------------------------------------------------------------------------------------------------
class Training(Inference):
    optimizer: any

    def __init__(self, *args, optimizer, **kwargs):
        super().__init__(*args, **kwargs)

        self.optimizer = optimizer


    def train(self, data_loader: DataLoader, loss: Loss, metrics: List[Metric]=[], desc=None):
        all_metrics = [loss] + metrics

        for metric in all_metrics:
            metric.start(len(data_loader.dataset), len(data_loader), self.num_classes)

        self.model.train()

        for idx, batch in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=desc if desc is not None else f'Training "{ self.name }"'):
            self.optimizer.zero_grad(set_to_none=True)

            batch = self.preprocess_batch_hook(batch)

            input = batch['inputs'].to('cuda')
            output = self.model(input)

            for metric in all_metrics:
                metric.update(self.model, input, output, batch)

            loss.backward(self.model, input, output, batch)

            self.optimizer.step()


#-----------------------------------------------------------------------------------------------------------------------
class TrainingWithCutmixMixup(Training):
    transform: any

    # pass cutmix=False / mixup=False to disable cutmix/mixup
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
        batch['images'], batch['labels'] = self.transform(batch['images'], batch['labels'])

        return super().preprocess_batch_hook(batch)
