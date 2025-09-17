from __future__ import annotations
from typing import List, Dict

import torch
import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
from transformers import AutoImageProcessor

from lib.checkpoint import CheckpointStorage, get_rng_state, set_rng_state
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
    def __init__(self, *args, bigger_is_better=False, patience=3, **kwargs):
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

            return True
        else:
            self.counter += 1

            return False

    @property
    def left(self):
        return self.patience - self.counter

    def should_stop(self):
        return self.counter >= self.patience



#-----------------------------------------------------------------------------------------------------------------------
class Trainer():
    name: str
    num_classes: int
    model_id: str
    seed: int
    checkpoint_storage: CheckpointStorage

    num_epochs: int

    epochs : List[Epoch]

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
        num_epochs=1, num_classes, max_epochs=15,
        model_id, model_preprocessor=None,
        seed=None, checkpoint_storage, training_cls=Training,
        dataloader_train=None, dataloader_val=None,
        loss=None, optimization_metric=None, metrics=[],
        early_stopping=None,
        has_saved_state=False,
        **kwargs
    ):
        super().__init__()

        self.name = name
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.max_epochs = max_epochs
        self.model_id = model_id
        self.training_cls = training_cls
        self.seed = seed
        self.checkpoint_storage = checkpoint_storage

        self.loss = loss
        self.optimization_metric = optimization_metric
        self.metrics = metrics

        self.epochs = []

        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        if not has_saved_state:
            self.training = self.create_training()
            self.early_stopping = early_stopping


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

    def __getstate__(self):
        return {
            'name': self.name,
            'max_epochs': self.max_epochs,
            'num_classes': self.num_classes,
            'model_id': self.model_id,
            'training': self.training,
            'early_stopping': self.early_stopping,
            'epochs': self.epochs,
            'rng' : get_rng_state()
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        if state['rng']:
            set_rng_state(state['rng'])


    @classmethod
    def load_or_create(cls, *args, checkpoint_storage, **kwargs):
        filename, epoch = checkpoint_storage.latest(r'checkpoint_(\d+).pth')

        instance = cls(*args, **{**kwargs, 'checkpoint_storage': checkpoint_storage, 'has_saved_state': filename is not None})

        if filename is not None:
            instance.__setstate__(torch.load(filename, weights_only=False))

        return instance


    def save(self, epoch):
        self.checkpoint_storage.touch()
        torch.save(self.__getstate__(), self.checkpoint_storage.dir / f'checkpoint_{ str(epoch.idx).rjust(3, "0") }.pth')


    def resume(self):
        return self.fit()


    def start_epoch(self, idx):
        print(f'Starting epoch {idx}')

        return Epoch(idx=idx)


    def finish_epoch(self, epoch):
        self.epochs.append(epoch)

        self.save(epoch)


    def fit(self):
        assert self.training is not None

        validation_metric = self.optimization_metric or self.loss

        metrics_with_loss = list(filter(lambda el: bool(el), [ self.loss, self.optimization_metric, *self.metrics ]))
        metrics = [validation_metric, *self.metrics]

        epoch_start = self.epochs[-1].idx + 1 if len(self.epochs) > 0 else 0
        epoch_end = epoch_start + self.num_epochs

        if epoch_start >= self.max_epochs:
            print(f'Already trained for maximum number of epochs {self.max_epochs}')
            return
        if epoch_end > self.max_epochs:
            epoch_end = self.max_epochs

        if self.early_stopping.best_score_at is not None:
            print(f'Early stopping: best model at epoch {self.early_stopping.best_score_at} with score {self.early_stopping.best_score}, {self.early_stopping.left} epochs left')

        for epoch_idx in range(epoch_start, epoch_end):
            epoch = self.start_epoch(epoch_idx)

            # Training
            self.training.train(self.dataloader_train, metrics=metrics_with_loss)

            for metric in self.metrics:
                epoch.push('traininig/' + metric.id, metric.value())

            print(f'Train: Loss {self.loss.name}={self.loss.value():.3f}{ '' if self.optimization_metric is None else f', metric {self.optimization_metric.name}={self.optimization_metric.value()}'}')

            # Validation
            self.training.infer(self.dataloader_val, metrics=metrics, desc=f'Validation { self.name }')

            for metric in self.metrics:
                epoch.push('validation/' + metric.id, metric.value())
                print(f'Validation: {metric.name}={metric.value():.3f}')

            print(f'Validation: Loss {self.loss.name}={self.loss.value():.3f}{ '' if self.optimization_metric is None else f', metric {self.optimization_metric.name}={self.optimization_metric.value()}'}')

            is_better = self.early_stopping.push(validation_metric.value())

            self.finish_epoch(epoch)

            if is_better:
                print(f'Early stopping: found better model at epoch {epoch_idx}')

            if self.early_stopping.should_stop():
                print(f'Early stopping: {self.early_stopping.best_score} at epoch {self.early_stopping.best_score_at}')
                break
            else:
                print(f'Early stopping: {self.early_stopping.left} epochs left')

