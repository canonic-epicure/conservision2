import argparse
from pathlib import Path

from lib.checkpoint import CheckpointStorage
from lib.training import Training

parser = argparse.ArgumentParser(description='KFold training')

parser.add_argument('--checkpoint_dir', default='checkpoints_convnext_large', type=str, help='directory for checkpoints')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=15, type=int)
parser.add_argument('--num_folds', default=5, type=int)
parser.add_argument('--num_workers', default=6, type=int, help='number of workers for prefetcing the data')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)

args = parser.parse_args()

model_id = "facebook/convnext-large-384-22k-1k"
data_dir = Path(__file__) / "data"


checkpoint_dir = Path(args.checkpoint_dir)

checkpoint_storage = CheckpointStorage(dir=checkpoint_dir)

training = Training(
    checkpoint_dir=args.checkpoint_dir,
    seed=args.seed,
    num_gpus=1,
    gpu_id=args.gpuid,
)

training.resume()