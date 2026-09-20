import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../.."
    )
)

sys.path.append(PROJECT_ROOT)


from src.anatomy.datasets.optic_disc_dataset import (
    OpticDiscDataset,
    get_idrid_optic_disc_samples
)

from src.anatomy.models.optic_disc_unet import (
    OpticDiscUNet
)


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "anatomy"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "anatomy",
    "optic_disc"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_optic_disc_unet.pth"
)

IMAGE_SIZE = (512, 512)

BATCH_SIZE = 2

EPOCHS = 30

LR = 1e-4

VAL_SPLIT = 0.20

SEED = 42

NUM_WORKERS = 0


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        targets
    ):

        probs = torch.sigmoid(
            logits
        )

        intersection = (
            probs * targets
        ).sum(
            dim=(1, 2, 3)
        )

        denominator = (
            probs.sum(dim=(1, 2, 3))
            +
            targets.sum(dim=(1, 2, 3))
        )

        dice = (
            2 * intersection
            +
            self.smooth
        ) / (
            denominator
            +
            self.smooth
        )

        return 1 - dice.mean()


# ============================================================
# BCE + DICE
# ============================================================

class BCEDiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(
        self,
        logits,
        targets
    ):

        return (
            self.bce(logits, targets)
            +
            self.dice(logits, targets)
        )


# ============================================================
# DICE SCORE
# ============================================================

def dice_score(
    logits,
    targets
):

    probs = torch.sigmoid(
        logits
    )

    predictions = (
        probs > 0.5
    ).float()

    intersection = (
        predictions * targets
    ).sum(
        dim=(1, 2, 3)
    )

    denominator = (
        predictions.sum(dim=(1, 2, 3))
        +
        targets.sum(dim=(1, 2, 3))
    )

    dice = (
        2 * intersection + 1e-7
    ) / (
        denominator + 1e-7
    )

    return dice.mean().item()


# ============================================================
# IOU
# ============================================================

def iou_score(
    logits,
    targets
):

    probs = torch.sigmoid(
        logits
    )

    predictions = (
        probs > 0.5
    ).float()

    intersection = (
        predictions * targets
    ).sum(
        dim=(1, 2, 3)
    )

    union = (
        predictions.sum(dim=(1, 2, 3))
        +
        targets.sum(dim=(1, 2, 3))
        -
        intersection
    )

    iou = (
        intersection + 1e-7
    ) / (
        union + 1e-7
    )

    return iou.mean().item()


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("OPTIC DISC SEGMENTATION")
print("EfficientNet-B0 + U-Net")
print("=" * 70)

print("\nDevice:", DEVICE)


# ============================================================
# LOAD DATA
# ============================================================

samples = get_idrid_optic_disc_samples(
    DATA_ROOT
)

if len(samples) == 0:

    raise RuntimeError(
        "No Optic Disc samples found."
    )


dataset = OpticDiscDataset(
    samples,
    IMAGE_SIZE
)


# ============================================================
# SPLIT
# ============================================================

val_size = int(
    len(dataset) * VAL_SPLIT
)

train_size = (
    len(dataset)
    - val_size
)

generator = torch.Generator().manual_seed(
    SEED
)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

print("\nTrain:", len(train_dataset))
print("Val  :", len(val_dataset))


# ============================================================
# LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# MODEL
# ============================================================

model = OpticDiscUNet().to(
    DEVICE
)

criterion = BCEDiceLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-5
)


# ============================================================
# TRAIN
# ============================================================

best_dice = 0.0

patience = 7

no_improvement = 0


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0

    for images, masks in tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS} TRAIN"
    ):

        images = images.to(
            DEVICE
        )

        masks = masks.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()


    train_loss /= len(
        train_loader
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0

    val_dice = 0

    val_iou = 0

    with torch.no_grad():

        for images, masks in tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} VAL"
        ):

            images = images.to(
                DEVICE
            )

            masks = masks.to(
                DEVICE
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                masks
            )

            val_loss += loss.item()

            val_dice += dice_score(
                outputs,
                masks
            )

            val_iou += iou_score(
                outputs,
                masks
            )


    val_loss /= len(
        val_loader
    )

    val_dice /= len(
        val_loader
    )

    val_iou /= len(
        val_loader
    )


    print("\n" + "-" * 70)

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss  : {val_loss:.4f}"
    )

    print(
        f"Val Dice  : {val_dice:.4f}"
    )

    print(
        f"Val IoU   : {val_iou:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST
    # --------------------------------------------------------

    if val_dice > best_dice:

        best_dice = val_dice

        no_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou
            },
            MODEL_PATH
        )

        print(
            "✓ BEST OPTIC DISC MODEL SAVED"
        )

    else:

        no_improvement += 1

        print(
            f"No improvement "
            f"({no_improvement}/{patience})"
        )


    if no_improvement >= patience:

        print(
            "\nEarly stopping."
        )

        break


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)

print("OPTIC DISC TRAINING COMPLETE")

print("=" * 70)

print(
    f"Best Val Dice: {best_dice:.4f}"
)

print(
    f"Model: {MODEL_PATH}"
)