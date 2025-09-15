from __future__ import annotations
from typing import List, Dict

import torch
import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
from transformers import AutoImageProcessor

from lib.checkpoint import CheckpointStorage
from lib.metric import Metric, Loss, ProbabilitiesMetric, CrossEntropyLoss, AccuracyMetric
from lib.training import Training

#-----------------------------------------------------------------------------------------------------------------------
class Epoch():
    idx: int

    metrics : Dict[str, any] = {}

    def __init__(self, *args, idx=0,  **kwargs):
        self.idx = idx
        self.metrics = {}

    def push(self, id, value):
        self.metrics[id] = value




#-----------------------------------------------------------------------------------------------------------------------
class EarlyStopping():
    def __init__(self, *args, bigger_is_better=False, patience=5, **kwargs):
        super().__init__(*args, **kwargs)

        self.patience = patience
        self.bigger_is_better = bigger_is_better

        self.counter = 0
        self.best_score = None
        self.best_score_at = None
        self.tracking = []

    def is_better(self, score_prev, score_new):
        if self.bigger_is_better:
            return score_new > score_prev
        else:
            return score_new < score_prev

    def push(self, value):
        self.tracking.append(value)

        if (self.best_score is None) or self.is_better(self.best_score, value):
            self.best_score = value
            self.best_score_at = len(self.tracking) - 1
            self.counter = 0
        else:
            self.counter += 1

    def should_stop(self):
        return self.counter >= self.patience



#-----------------------------------------------------------------------------------------------------------------------
class Trainer():
    name: str
    num_classes: int
    model_id: str
    model_preprocessor: any
    seed: int
    checkpoint_storage: CheckpointStorage

    num_epochs: int

    epochs : List[Epoch]
    current_epoch: Epoch

    dataloader_train: any
    dataloader_val: any

    loss : Loss
    optimization_metric : Metric
    metrics: List[Metric]


    training_cls : any
    training : Training


    def __init__(
        self,
        *args,
        name='Unknown trainer',
        num_epochs=15, num_classes,
        model_id, model_preprocessor=None,
        seed=None, checkpoint_storage, training_cls=Training,
        dataloader_train=None, dataloader_val=None,
        patience=5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.name = name
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.model_id = model_id
        self.model_preprocessor = model_preprocessor if model_preprocessor is not None else AutoImageProcessor.from_pretrained(model_id)
        self.training_cls = training_cls
        self.seed = seed
        self.checkpoint_storage = checkpoint_storage

        self.current_epoch = None
        self.epochs = []

        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        self.training = self.create_training()
        self.early_stopping = EarlyStopping(patience=patience)


    def create_model(self):
        raise NotImplementedError

    def create_optimizer(self, model):
        raise NotImplementedError


    def create_training(self):
        model = self.create_model()
        optimizer = self.create_optimizer(model)

        return self.training_cls(
            name=self.name,
            model=model,
            optimizer=optimizer,
            num_classes=self.num_classes
        )

    @classmethod
    def load_or_create(cls, *args, checkpoint_storage, **kwargs):
        filename, epoch = checkpoint_storage.latest(r'checkpoint_(\d+).pth')

        if filename is None:
            return cls(*args, **{**kwargs, 'checkpoint_storage': checkpoint_storage})
        else:
            return torch.load(filename)


    def save(self, epoch):
        self.checkpoint_storage.touch()
        torch.save(self, self.checkpoint_storage.dir / f'checkpoint_{ str(epoch.idx).rjust(3, "0") }.pth')


    def resume(self):
        return self.fit()


    def start_epoch(self, idx):
        print(f'Starting epoch {idx}')

        self.current_epoch = Epoch(idx=idx)

        return self.current_epoch


    def finish_epoch(self):
        epoch = self.current_epoch

        self.epochs.append(epoch)
        self.current_epoch = None

        self.save(epoch)


    def fit(self, epochs_to_train=1):
        assert self.training is not None

        metrics_with_loss = filter(lambda el: bool(el), [ self.loss, self.optimization_metric, *self.metrics ])
        metrics = filter(lambda el: bool(el), [self.optimization_metric or self.loss, *self.metrics])

        epoch_start = self.epochs[-1].idx + 1 if len(self.epochs) > 0 else 0
        epoch_end = max(self.epoch_start + epochs_to_train, self.num_epochs)

        for epoch_idx in range(epoch_start, epoch_end):
            epoch = self.start_epoch(epoch_idx)

            # Training
            self.training.train_one_cycle(self.dataloader_train, metrics=metrics_with_loss)

            for metric in self.metrics:
                epoch.push('traininig/' + metric.id, metric.value())

            print(f'Train: loss={self.loss.value()}{ '' if self.optimization_metric is None else f' Optimization metric={self.optimization_metric.value()}'}')

            # Validation
            self.training.infer(self.dataloader_val, metrics=metrics)

            for metric in self.metrics:
                epoch.push('validation/' + metric.id, metric.value())
                print(f'Validation: {metric.name}={metric.value()}')

            print(f'Validation: metric={(self.optimization_metric or self.loss).value()}')

            self.finish_epoch(epoch_idx)

            self.early_stopping.push((self.optimization_metric or self.loss).value())

            if self.early_stopping.should_stop():
                print(f'Early stopping: {self.early_stopping.best_score} at epoch {self.early_stopping.best_score_at}')
                break
