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
    os.path.join(os.path.dirname(__file__), "../../..")
)

sys.path.append(PROJECT_ROOT)

from src.anatomy.datasets.vessel_dataset import (
    VesselDataset,
    get_all_vessel_samples
)

from src.anatomy.models.unet_efficientnet import VesselUNet


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "anatomy")

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "anatomy",
    "vessels"
)

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_vessel_unet.pth"
)

IMAGE_SIZE = (512, 512)

BATCH_SIZE = 2
NUM_EPOCHS = 30

LEARNING_RATE = 1e-4

VAL_SPLIT = 0.20

RANDOM_SEED = 42

NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("=" * 70)
print("RETINAL BLOOD VESSEL SEGMENTATION")
print("EfficientNet-B0 + U-Net")
print("=" * 70)

print(f"\nDevice: {DEVICE}")

if DEVICE.type == "mps":
    print("Apple Silicon GPU acceleration: ENABLED")


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.contiguous()
        targets = targets.contiguous()

        intersection = (probabilities * targets).sum(
            dim=(1, 2, 3)
        )

        denominator = (
            probabilities.sum(dim=(1, 2, 3))
            +
            targets.sum(dim=(1, 2, 3))
        )

        dice = (
            (2.0 * intersection + self.smooth)
            /
            (denominator + self.smooth)
        )

        return 1.0 - dice.mean()


# ============================================================
# COMBINED LOSS
# ============================================================

class BCEDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):

        bce_loss = self.bce(logits, targets)

        dice_loss = self.dice(logits, targets)

        total_loss = bce_loss + dice_loss

        return total_loss


# ============================================================
# DICE SCORE
# ============================================================

def dice_score(logits, targets, threshold=0.5):

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities > threshold).float()

    predictions = predictions.contiguous()
    targets = targets.contiguous()

    intersection = (predictions * targets).sum(
        dim=(1, 2, 3)
    )

    denominator = (
        predictions.sum(dim=(1, 2, 3))
        +
        targets.sum(dim=(1, 2, 3))
    )

    dice = (
        (2.0 * intersection + 1e-7)
        /
        (denominator + 1e-7)
    )

    return dice.mean().item()


# ============================================================
# IOU SCORE
# ============================================================

def iou_score(logits, targets, threshold=0.5):

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities > threshold).float()

    predictions = predictions.contiguous()
    targets = targets.contiguous()

    intersection = (predictions * targets).sum(
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
        (intersection + 1e-7)
        /
        (union + 1e-7)
    )

    return iou.mean().item()


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading vessel dataset...")

samples = get_all_vessel_samples(DATA_ROOT)

print(f"\nTotal samples: {len(samples)}")

if len(samples) == 0:

    raise RuntimeError(
        "No vessel samples found. Check your dataset paths."
    )


# ============================================================
# DATASET
# ============================================================

dataset = VesselDataset(
    samples,
    image_size=IMAGE_SIZE
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

total_size = len(dataset)

val_size = int(total_size * VAL_SPLIT)

train_size = total_size - val_size

generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

print("\nDataset split:")
print(f"Training samples   : {len(train_dataset)}")
print(f"Validation samples : {len(val_dataset)}")


# ============================================================
# DATALOADERS
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

print("\nCreating model...")

model = VesselUNet()

model = model.to(DEVICE)

print("Model created successfully.")


# ============================================================
# LOSS
# ============================================================

criterion = BCEDiceLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


# ============================================================
# TRAINING
# ============================================================

best_val_dice = 0.0

epochs_without_improvement = 0

EARLY_STOPPING_PATIENCE = 7


print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print(f"Epochs       : {NUM_EPOCHS}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Image size   : {IMAGE_SIZE}")
print("=" * 70)


for epoch in range(NUM_EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_dice = 0.0

    train_progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [TRAIN]"
    )

    for images, masks in train_progress:

        images = images.to(
            DEVICE,
            dtype=torch.float32
        )

        masks = masks.to(
            DEVICE,
            dtype=torch.float32
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        batch_dice = dice_score(
            outputs.detach(),
            masks
        )

        train_loss += loss.item()
        train_dice += batch_dice

        train_progress.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{batch_dice:.4f}"
        )


    train_loss /= len(train_loader)
    train_dice /= len(train_loader)


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0

    with torch.no_grad():

        val_progress = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [VAL]"
        )

        for images, masks in val_progress:

            images = images.to(
                DEVICE,
                dtype=torch.float32
            )

            masks = masks.to(
                DEVICE,
                dtype=torch.float32
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            dice = dice_score(
                outputs,
                masks
            )

            iou = iou_score(
                outputs,
                masks
            )

            val_loss += loss.item()
            val_dice += dice
            val_iou += iou


    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    val_iou /= len(val_loader)


    # --------------------------------------------------------
    # LEARNING RATE
    # --------------------------------------------------------

    scheduler.step(val_dice)

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n")
    print("-" * 70)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Train Dice : {train_dice:.4f}"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Dice   : {val_dice:.4f}"
    )

    print(
        f"Val IoU    : {val_iou:.4f}"
    )

    print(
        f"Learning Rate: {current_lr:.7f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_dice > best_val_dice:

        best_val_dice = val_dice

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou
            },
            MODEL_PATH
        )

        print(
            f"\n✓ BEST MODEL SAVED"
        )

        print(
            f"  Val Dice: {val_dice:.4f}"
        )

        print(
            f"  Val IoU : {val_iou:.4f}"
        )

        print(
            f"  Path    : {MODEL_PATH}"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"\nNo improvement "
            f"({epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE})"
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

        print("\n" + "=" * 70)
        print("EARLY STOPPING")
        print("=" * 70)

        break


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best Validation Dice: {best_val_dice:.4f}"
)

print(
    f"Model saved at:\n{MODEL_PATH}"
)

print("=" * 70)