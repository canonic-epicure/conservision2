import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class DatasetWithLabel(Dataset):
    def __init__(self, *args, labels: pd.DataFrame=None, **kwargs):
        self.labels = labels
        super().__init__(*args, **kwargs)

    def get_label(self, idx):
        return torch.tensor(self.labels.iloc[idx].values).float()

    def __getitem__(self, idx):
        sup = super().__getitem__(idx)

        sup['labels'] = self.get_label(idx)

        return sup


class ImageDataset(Dataset):
    def __init__(self, *args, data: pd.Series, processor, aug=None, **kwargs):
        self.data = data

        self.processor = processor
        self.aug = aug

        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.data)

    def get_image(self, idx):
        image_id = self.data.index[idx]

        image = Image.open(self.data.iloc[idx]).convert("RGB")

        image = self.aug(image) if self.aug else image
        image = self.processor(image)

        return image, image_id

    def __getitem__(self, idx):
        sup = super().__getitem__(idx)

        image, image_id = self.get_image(idx)

        sup['images'] = image
        sup['ids'] = image_id

        return sup


class ImageDatasetWithLabel(DatasetWithLabel, ImageDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        return super().__getitem__(idx)