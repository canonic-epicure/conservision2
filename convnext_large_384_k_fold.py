from typing import Literal, Union

import argparse
from pathlib import Path
import math
from adamp import AdamP
from torchvision.transforms.v2.functional import to_pil_image

import matplotlib.pyplot as plt
import pandas as pd
import torch
import asyncio
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

import data
import lib
import lib.photo
import lib.dataset
import lib.transformations

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
model_preprocessor = AutoImageProcessor.from_pretrained(model_id)
data_dir = Path(__file__) / "data"

#-----------------------------------------------------------------------------------------------------------------------
for fold in range(args.num_folds):
    print(f"Fold {fold}")

    fold_mask = data.train_all['fold'] == fold

    train_data = data.train_all[~fold_mask]
    val_data = data.train_all[fold_mask]

    train_loader = DataLoader(
        lib.dataset.ImageDatasetWithLabel(data_dir=data_dir, data=train_data, labels=train_data['label'], processor=model_preprocessor, aug=lib.transformations.transform_training),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader   = DataLoader(
        lib.dataset.ImageDatasetWithLabel(data_dir=data_dir, data=val_data, labels=val_data['label'], processor=model_preprocessor, aug=lib.transformations.transform_inference),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )



def train_model(model, optimizer, scheduler, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch}")





#%%
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# criterion = lib.sce_loss
#%% md
# ### Cutmix + mixup
#%%
from torchvision.transforms import v2

use_cutmix_mixup = True

cutmix = v2.CutMix(alpha=0.3, num_classes=len(data.species_labels))
mixup = v2.MixUp(alpha=0.3, num_classes=len(data.species_labels))
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
#%%
# steps_per_epoch = len(train_loader)
#
# def scheduler(step):
#




for cur_epoch in range(epoch, epoch + num_epochs):
    await asyncio.sleep(0)

    if stop_flag['value'] == True:
        break

    print(f"Starting epoch {cur_epoch}")

    model.train()

    loss_acc = 0
    count = 0

    for idx, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc='Training'):
        optimizer.zero_grad(set_to_none=True)

        images, labels = batch["pixel_values"].to(torch.device("cuda")), batch["labels"].to(torch.device("cuda"))

        if use_cutmix_mixup:
            images, labels = cutmix_or_mixup(images, labels)

        refined_labels_df = model(images)              # logits: (B, 2)
        loss = criterion(refined_labels_df.logits, labels)

        c = batch['pixel_values'].size(0)
        loss_acc += loss.item() * c
        count += c

        loss.backward()
        optimizer.step()

    tracking_loss.append(loss_acc / count)

    # валидация
    model.eval()

    probs, loss_acc = predict_siglip(
        model, val_loader, accumulate_probs=True, accumulate_loss=True, desc='Validation', columns=data.species_labels, criterion=criterion
    )
    tracking_val_probs.append(probs)
    tracking_loss_val.append(loss_acc)

    eval_predictions = probs.idxmax(axis=1)
    eval_true = data.y_eval.idxmax(axis=1)
    correct = (eval_predictions == eval_true).sum()
    accuracy = correct / len(eval_predictions)
    tracking_accuracy.append(accuracy.item())

    model.epoch = cur_epoch
    lib.save_model(model, optimizer, f"./models_convnext_large_384/checkpoint_{str(cur_epoch).rjust(2, "0")}.pth")

    epoch = cur_epoch + 1

# #%% md
# # ## Training progress
# #%%
# pd.DataFrame({'tracking_loss' : tracking_loss, 'tracking_loss_val' : tracking_loss_val, 'tracking_accuracy' : tracking_accuracy }, index=range(len(tracking_accuracy)))
# #%%
# fig, ax = plt.subplots(figsize=(15, 5))
#
# epochs_train = list(range(len(tracking_loss)))
# epochs_val = list(range(len(tracking_loss_val)))
#
# line1, = ax.plot(epochs_train, tracking_loss, label="Train loss")
# line2, = ax.plot(epochs_val, tracking_loss_val, label="Validation loss")
#
# ax.set_xlabel("Epoch (index)")
# ax.set_ylabel("Loss")
# ax.legend(loc="best", handles=[line1, line2])
#
# ax.set_xticks(epochs_train)
#
# ax.grid(True)
# #%%
# fig, ax = plt.subplots(figsize=(15, 5))
#
# epochs_accuracy = list(range(len(tracking_accuracy)))
#
# line1, = ax.plot(epochs_accuracy, tracking_accuracy, label="Accuracy", color="red")
# ax.set_ylabel("Accuracy")
#
# ax.legend(loc="best", handles=[line1])
#
# ax.set_xticks(epochs_train)
#
# ax.grid(True)
# #%% md
# # ## Validation
# #%%
# ## search for optimal temperature
# #%%
# # temp_acc = {}
# #%%
# # for key in sorted(temp_acc.keys()):
# #     print(f'T={key:.5f}: {temp_acc[key]:.5f}')
# #%%
# # import numpy as np
# #
# # for t in np.arange(0.785, 0.82, 0.0125):
# #     _, loss = lib.predict_siglip(model, val_loader, accumulate_loss=True, accumulate_probs=False, criterion=criterion, T=t, desc='Searching', columns=data.species_labels)
# #
# #     print(f"T={t:.5f}: {loss:.4f}")
# #
# #     temp_acc[t] = loss
# #%%
# eval_preds_df = tracking_val_probs[-1]
#
# # eval_preds_df_ten_crop = lib.predict_siglip_ten_crop(model, val_loader, T=1, desc='Predicting', columns=data.species_labels)
# #%%
# eval_preds_df.head()
# #%%
# # eval_preds_df_ten_crop.head()
# #%%
# print("True labels (training):")
# data.y_train.idxmax(axis=1).value_counts(normalize=True)
# #%%
# print("Predicted labels (eval):")
# eval_preds_df.idxmax(axis=1).value_counts(normalize=True)
# #%%
# print("True labels (eval):")
# data.y_eval.idxmax(axis=1).value_counts(normalize=True)
# #%%
# eval_predictions = eval_preds_df.idxmax(axis=1)
# # eval_predictions_ten_crop = eval_preds_df_ten_crop.idxmax(axis=1)
# eval_true = data.y_eval.idxmax(axis=1)
# #%%
# # (eval_predictions_ten_crop != eval_predictions).sum()
# #%%
# print(f'Accuracy plain: { (eval_predictions == eval_true).mean() }')
# # print(f'Accuracy ten crop: { (eval_predictions_ten_crop == eval_true).mean() }')
# #%% md
# # ### Predictions vs actual
# #%%
# eval_preds = eval_preds_df.copy()
#
# eval_preds[ 'cls' ] = eval_preds_df.idxmax(axis=1)
# eval_preds[ 'cls_true' ] = data.y_eval.idxmax(axis=1)
#
# # eval_preds[(eval_preds[ 'cls' ] == 'blank') & (eval_preds[ 'cls_true' ] == 'leopard')]
# #%%
# data.species_labels
# #%%
# import math
# from itertools import zip_longest
# from PIL import Image
# %matplotlib inline
# # %matplotlib notebook
# # %matplotlib widget
# from torchvision.transforms.functional import to_pil_image
#
# random_state = 41111
#
# # rows = eval_preds[(eval_preds[ 'cls' ] == 'blank') & (eval_preds[ 'cls_true' ] == 'leopard')]
# rows = eval_preds[(eval_preds[ 'cls' ] == 'blank') & (eval_preds[ 'cls_true' ] == 'leopard')]
#
# rows = rows.sample(frac=0.2, random_state=random_state)
#
# n_cols = 3
# n_rows = math.ceil(len(rows) / n_cols)
#
# fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 35))
#
# # fig.canvas.layout.width = '100%'   # ширину займёт вся ячейка
# # Высоту ipywidgets не умеют «auto», укажи соотношение/высоту:
# # fig.set_figheight(fig.get_figwidth() * 0.6)
#
# invert = v2.RandomInvert(p=1)
#
# # iterate through each species
# print(f'Total rows: {len(rows)}')
#
# clahe = lib.LabCLAHE()
#
# for row, ax in zip_longest(list(rows.iterrows()), axes.flatten()):
#     if row is None:
#         if ax is not None:
#             ax.remove()
#         continue
#     if ax is None:
#         break
#     img = Image.open('data/train_features/' + row[0] + '.jpg')
#     ax.imshow(to_pil_image(clahe(clahe((img)))))
#     ax.set_title(f"{row[1].name} ")
#
# fig.tight_layout()
# #%% md
# # ### Confusion matrix
# #%%
# from sklearn.metrics import ConfusionMatrixDisplay
#
# fig, ax = plt.subplots(figsize=(10, 10))
# cm = ConfusionMatrixDisplay.from_predictions(
#     data.y_eval.idxmax(axis=1),
#     eval_preds_df.idxmax(axis=1),
#     ax=ax,
#     xticks_rotation=30,
#     colorbar=True,
# )
# #%% md
# # ## Create submission
# #%%
# test_dataset = lib.ImageDatasetSigLip2(data.test_features, processor=model_preprocessor, learning=False)
# test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=6)
# #%%
# submission_df, _ = predict_siglip(model, test_dataloader, T=1, columns=data.species_labels)
# #%%
# submission_format = pd.read_csv("data/submission_format.csv", index_col="id")
#
# assert all(submission_df.index == submission_format.index)
# assert all(submission_df.columns == submission_format.columns)
# #%%
# submission_df.to_csv("submission.csv")