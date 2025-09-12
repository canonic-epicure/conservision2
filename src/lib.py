import glob
import re
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

import random




def background_template(img_batch: torch.Tensor) -> torch.Tensor:
    assert img_batch.ndim == 4

    return img_batch.median(dim=0).values


def affine_params_to_background(x_i: torch.Tensor, background_i: torch.Tensor, eps:float =1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x_i.shape == background_i.shape, "x и background должны совпадать по форме."

    x = x_i
    background = background_i

    # Средние и СКО по всем пикселям каждого канала
    mux  = x.mean(dim=(1, 2))
    sdx  = x.std(dim=(1, 2), unbiased=False).clamp_min(eps)
    muB  = background.mean(dim=(1, 2))
    sdB  = background.std(dim=(1, 2), unbiased=False).clamp_min(eps)

    alpha = sdB / sdx            # (C,)
    beta  = muB - alpha * mux    # (C,)

    return alpha, beta

def similarity(img1_u8: torch.Tensor, img2_u8: torch.Tensor) -> float:
    img1_u8 = to_rgb(img1_u8)
    img2_u8 = to_rgb(img2_u8)

    assert img1_u8.dtype == torch.uint8 and img1_u8.ndim == 3 and img1_u8.shape[0] == 3
    assert img2_u8.dtype == torch.uint8 and img2_u8.ndim == 3 and img2_u8.shape[0] == 3

    C, H1, W1 = img1_u8.shape
    C, H2, W2 = img2_u8.shape

    assert H1 == H2 and W1 == W2




class RandomInvertIfGrayscale:
    def __init__(self, p=0.5):
        self.p = p
        self.invert = v2.RandomInvert(p=1.0)

    @torch.no_grad()
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("img must be PIL.Image ")

        is_grayscale, _ = photo.looks_grayscale_ycbcr_cv(img)

        # no-op if image is not grayscale or random.random() > self.p
        if not is_grayscale or random.random() > self.p:
            return img

        return self.invert(img)


class ImagesDatasetResnet(Dataset):
    def __init__(self, x_df, y_df=None, learning=True):
        self.data = x_df
        self.label = y_df

        self.transform = v2.Compose(
            [
                LabCLAHE(),
                LabCLAHE(),
                v2.ToPILImage(),

                v2.ColorJitter() if learning else lambda x: x,
                v2.RandomAutocontrast() if learning else lambda x: x,
                v2.RandomEqualize() if learning else lambda x: x,
                v2.RandomAdjustSharpness(sharpness_factor=1.5) if learning else lambda x: x,
                v2.RandomHorizontalFlip() if learning else lambda x: x,
                v2.RandomRotation(degrees=15, interpolation=InterpolationMode.BICUBIC) if learning else lambda x: x,

                v2.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __getitem__(self, index):
        image = Image.open('data/' + self.data.iloc[index]["filepath"]).convert("RGB")
        image = self.transform(image)
        image_id = self.data.index[index]
        # if we don't have labels (e.g. for test set) just return the image and image id
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label = torch.tensor(self.label.iloc[index].values, dtype=torch.float)
            sample = {"image_id": image_id, "image": image, "label": label}
        return sample

    def __len__(self):
        return len(self.data)


siglip2_training_transform = v2.Compose(
    [
        LabCLAHE(),
        LabCLAHE(),
        v2.ToPILImage(),

        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
        v2.RandomAutocontrast(p=0.1),
        v2.RandomZoomOut(p=0.1),
        v2.RandomEqualize(p=0.1),
        v2.RandomAdjustSharpness(p=0.3, sharpness_factor=1.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15, interpolation=InterpolationMode.BICUBIC),
    ]
)
siglip2_inference_transform = v2.Compose(
    [
        LabCLAHE(),
        LabCLAHE(),
        v2.ToPILImage(),
    ]
)

class ImageDatasetSigLip2(Dataset):
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame=None, processor=None, learning=True):
        self.data = data
        self.labels = labels
        self.processor = processor
        self.learning = learning

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img = Image.open('data/' + self.data.iloc[idx]["filepath"]).convert("RGB")
        image_id = self.data.index[idx]

        if self.learning:
            img = siglip2_training_transform(img)
        else:
            img = siglip2_inference_transform(img)

        # enc["pixel_values"]: (1, C, H, W) -> уберём размерность 0
        enc = self.processor(images=img, return_tensors="pt")
        if self.labels is None:
            return {
                "image_id": image_id,
                "pixel_values": enc["pixel_values"].squeeze(0)
            }
        else:
            return {
                "image_id": image_id,
                "pixel_values": enc["pixel_values"].squeeze(0),
                "labels": torch.tensor(self.labels.iloc[idx].values, dtype=torch.float)
            }

    # def collate_fn(batch):
    #     pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    #     labels = torch.stack([b["labels"] for b in batch], dim=0)
    #     return {"pixel_values": pixel_values, "labels": labels}


def save_model(model, optimizer, file_name: str):
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)

    torch.save({'model' : model, 'optimizer': optimizer}, file_name)

def model_checkpoints(glob_pattern: str) -> List[str]:
    files = glob.glob(glob_pattern)

    pattern = re.compile(r"checkpoint_(\d+)\.pth$")

    epochs = [ pattern.search(file)[1] for file in files if pattern.search(file) != None ]

    epochs.sort(reverse=True)

    return epochs


def predict_siglip(model, data_loader: DataLoader, accumulate_probs=True, accumulate_loss=False, T=1, desc='Predicting', criterion=None, columns=None):
    preds_collector = []

    # put the model in eval mode so we don't update any parameters
    model.eval()

    model.to(torch.device("cuda"))

    loss_acc = 0
    count = 0

    # we aren't updating our weights so no need to calculate gradients
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, total=len(data_loader), desc=desc):
            # 1) run the forward step
            logits = model.forward(batch["pixel_values"].to(torch.device("cuda"))).logits / T

            if accumulate_loss:
                loss = criterion(logits, batch["labels"].to('cuda'))

                c = batch['pixel_values'].size(0)
                loss_acc += loss.item() * c
                count += c

            if accumulate_probs:
                # 2) apply softmax so that model outputs are in range [0,1]
                preds = F.softmax(logits / T, dim=1)
                # 3) store this batch's predictions in df
                # note that PyTorch Tensors need to first be detached from their computational graph before converting to numpy arrays
                preds_df = pd.DataFrame(
                    preds.detach().to('cpu').numpy(),
                    index=batch["image_id"],
                    columns=columns,
                )
                preds_collector.append(preds_df)

    return pd.concat(preds_collector) if accumulate_probs else None, loss_acc / count if accumulate_loss else None



def predict_siglip_ten_crop(model, data_loader: DataLoader, T=1, desc='Predicting', columns=None):
    preds_collector = []

    # put the model in eval mode so we don't update any parameters
    model.eval()

    model.to(torch.device("cuda"))

    # we aren't updating our weights so no need to calculate gradients
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, total=len(data_loader), desc=desc):
            def process_image(image):
                w, h = image.shape[1:]

                resize = v2.Resize(size=(round(w * 1.5), round(h * 1.5)), interpolation=v2.InterpolationMode.BICUBIC)
                ten_crop = v2.TenCrop(size=(w, h))

                ten_crop_img = torch.stack(list(ten_crop(resize(image))))

                # 1) run the forward step
                logits = model.forward(ten_crop_img.to(torch.device('cuda'))).logits / T

                logp = F.log_softmax(logits, dim=1)  # (M, K)
                logp_mean = torch.logsumexp(logp, dim=0) - torch.log(torch.tensor(ten_crop_img.size(0)))  # (K,)

                return torch.exp(logp_mean).detach().to('cpu').numpy()

            process_image(batch["pixel_values"][0])

            preds_df = pd.DataFrame(
                [ process_image(image) for image in batch["pixel_values"] ],
                index=batch["image_id"],
                columns=columns,
            )
            preds_collector.append(preds_df)

    return pd.concat(preds_collector)


def sce_loss(logits, target, alpha=0.1, beta=1.0, num_classes=None, eps=1e-4):
    ce = F.cross_entropy(logits, target, reduction='none')  # [N]

    # one-hot с клампом (можно подать сюда и mixup-таргеты [N,C] без клампа)
    if num_classes is None:
        num_classes = logits.size(1)

    target = target.clamp_min(eps)  # [N,C]

    p = F.softmax(logits, dim=1)
    rce = -(p * target.log()).sum(dim=1)  # [N]

    loss = alpha * ce + beta * rce
    return loss.mean()


# # Утилита: делаем функцию-множитель для LambdaLR
# def make_cosine_multiplier(total_steps, warmup_steps, floor_ratio):
#     """
#     total_steps: на сколько шагов растянуть спад; после этого держим floor.
#     warmup_steps: сколько шагов линейно разогревать от 0 до 1.
#     floor_ratio: доля от базового LR, ниже которой не падаем (например, 0.1 = 10%).
#     Возвращает f(step_index) -> scale в [floor_ratio, 1.0].
#     """
#     total_steps = max(1, int(total_steps))
#     warmup_steps = max(0, int(warmup_steps))
#     floor_ratio = float(floor_ratio)
#
#     def f(step):
#         # В WARNING: в PyTorch первый scheduler.step() делает last_epoch=0.
#         # Поэтому step здесь – это last_epoch от LambdaLR.
#         if step < warmup_steps:
#             # линейный разогрев: 0 -> 1 (но не больше 1)
#             return (step + 1) / max(1, warmup_steps)
#         # прогресс внутри "косинусной" части [0..1]
#         t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
#         if t >= 1.0:
#             # вышли за горизонт: держим пол
#             return floor_ratio
#         # косинусный спад из [1..floor_ratio]
#         return floor_ratio + (1.0 - floor_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))
#     return f