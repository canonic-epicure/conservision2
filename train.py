import argparse
from pathlib import Path

import torch.optim
from torch.utils.data import DataLoader
from torch.utils.hipify.hipify_python import preprocessor
from transformers import AutoImageProcessor, AutoModelForImageClassification

from lib.checkpoint import CheckpointStorage, set_seed
from lib.dataset import ImageDatasetWithLabel
from lib.metric import CrossEntropyLoss, AccuracyMetric
from lib.trainer import Trainer
from lib.training import Training

from torchvision.transforms import v2, InterpolationMode

import lib.transformations
import src.data

parser = argparse.ArgumentParser(description='KFold training')

parser.add_argument('--checkpoint_dir', default='checkpoints_convnext_large', type=str, help='directory for checkpoints')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--max_epochs', default=15, type=int)
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--num_folds', default=5, type=int)
parser.add_argument('--num_workers', default=6, type=int, help='number of workers for prefetcing the data')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)

args = parser.parse_args('')

#-----------------------------------------------------------------------------------------------------------------------
set_seed(args.seed, True)

model_id = "timm/convnext_large.fb_in22k"
# model_id = "facebook/convnext-large-384-22k-1k"
# model_id = "google/siglip2-base-patch16-256"
data_dir = Path(__file__) / "data"
num_classes = len(src.data.species_labels)
preprocessor = AutoImageProcessor.from_pretrained(model_id)
def model_preprocessor(input):
    processed = preprocessor(images=input, return_tensors="pt")

    return processed['pixel_values'].squeeze(0)


#-----------------------------------------------------------------------------------------------------------------------
class ConvNextLargeTrainer(Trainer):
    def create_model(self):
        return AutoModelForImageClassification.from_pretrained(
            model_id,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        ).to('cuda')

    def create_optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)


#-----------------------------------------------------------------------------------------------------------------------
class ConvNextLargeTraining(Training):
    pass

import __main__
__main__.ConvNextLargeTraining = ConvNextLargeTraining


#-----------------------------------------------------------------------------------------------------------------------
trainer = ConvNextLargeTrainer.load_or_create(
    name="ConvNext large",
    num_classes=num_classes,
    num_epochs=args.num_epochs,
    max_epochs=args.max_epochs,
    model_id=model_id,
    checkpoint_storage=CheckpointStorage(dir=Path(args.checkpoint_dir)),
    training_cls=ConvNextLargeTraining,
    loss=CrossEntropyLoss(ce=torch.nn.CrossEntropyLoss(label_smoothing=0.1)),
    optimization_metric=None,
    metrics=[AccuracyMetric()],
    dataloader_train=DataLoader(
        ImageDatasetWithLabel(
            # data=src.data.train_all['filepath'].sample(n=50),
            # labels=src.data.train_all['label'],
            data=src.data.x_train['filepath'],#.sample(n=50),
            labels=src.data.x_train['label'],
            processor=model_preprocessor, aug=lib.transformations.transform_training
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    ),
    dataloader_val=DataLoader(
        ImageDatasetWithLabel(
            # data=src.data.train_all['filepath'].sample(n=50),
            # labels=src.data.train_all['label'],
            data=src.data.x_eval['filepath'],#.sample(n=50),
            labels=src.data.x_eval['label'],
            processor=model_preprocessor, aug=lib.transformations.transform_inference
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
)

if __name__ == "__main__":
    trainer.resume()