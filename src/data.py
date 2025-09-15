import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from PIL import Image
from pathlib import Path

data_dir = Path(__file__).parent / "../data/"

train_features = pd.read_csv(data_dir / "train_features.csv", index_col="id")
test_features = pd.read_csv(data_dir / "test_features.csv", index_col="id")
train_labels = pd.read_csv(data_dir / "train_labels.csv", index_col="id")

# verify one-hot encoding is correct
assert train_labels[train_labels.sum(axis=1) != 1].shape[0] == 0

species_labels = sorted(train_labels.columns.unique())

def get_resolution(filename):
    with Image.open(filename) as img:
        return f'{img.size[0]}x{img.size[1]}'

train_features['filepath'] = str(data_dir) + '/' + train_features['filepath']
test_features['filepath'] = str(data_dir) + '/' + test_features['filepath']

train_features['resolution'] = train_features['filepath'].apply(lambda filename: get_resolution(filename))

train_labels['label'] = train_labels.to_numpy().argmax(axis=1)

train_all = train_features.merge(train_labels[['label']], on='id')

train_all['fold'] = -1

splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(splitter.split(train_all, train_all['label'], groups=train_all['site'])):
    train_all.iloc[val_idx, train_all.columns.get_loc('fold')] = fold


# train_labels_refined = pd.read_csv("data/train_labels_refined.csv", index_col="id")
#
# assert all(train_labels.index == train_labels_refined.index)
# assert all(train_labels.columns == train_labels_refined.columns)

# assert train_labels_refined[train_labels_refined.sum(axis=1) != 1].shape[0] == 0
