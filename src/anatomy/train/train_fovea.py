import os
import sys
import torch
import numpy as np

from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.nn import SmoothL1Loss
from tqdm import tqdm


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../.."
    )
)

sys.path.append(PROJECT_ROOT)


from src.anatomy.datasets.fovea_dataset import FoveaDataset
from src.anatomy.models.fovea_net import FoveaNet


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "anatomy",
    "IDRid"
)

SAVE_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "anatomy",
    "fovea"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

SAVE_PATH = os.path.join(
    SAVE_DIR,
    "best_fovea_net.pth"
)

IMG_SIZE = 512

BATCH_SIZE = 4

EPOCHS = 30

LR = 1e-4

VAL_RATIO = 0.2

PATIENCE = 7


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


# ============================================================
# PIXEL ERROR
# ============================================================

def calculate_pixel_errors(
    predictions,
    targets
):

    predictions = (
        predictions
        .detach()
        .cpu()
        .numpy()
    )

    targets = (
        targets
        .detach()
        .cpu()
        .numpy()
    )

    difference = predictions - targets

    distances = np.sqrt(
        np.sum(
            difference ** 2,
            axis=1
        )
    )

    # Normalized coordinates → 512 × 512 pixels
    distances = distances * IMG_SIZE

    return distances


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("                 FOVEA LOCALIZATION TRAINING")
    print("=" * 70)

    print(f"Device       : {device}")
    print(f"Image size   : {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Epochs       : {EPOCHS}")
    print(f"Learning rate: {LR}")

    # ========================================================
    # DATASET
    # ========================================================

    print("\n" + "-" * 70)
    print("LOADING DATASET")
    print("-" * 70)

    dataset = FoveaDataset(
        DATA_ROOT
    )

    print(
        f"\nTotal Fovea samples: {len(dataset)}"
    )

    if len(dataset) < 5:

        raise RuntimeError(
            "Too few Fovea samples found."
        )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    val_size = int(
        len(dataset) * VAL_RATIO
    )

    train_size = (
        len(dataset)
        - val_size
    )

    generator = torch.Generator().manual_seed(
        42
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [
            train_size,
            val_size
        ],
        generator=generator
    )

    print(
        f"Train samples      : {len(train_dataset)}"
    )

    print(
        f"Validation samples : {len(val_dataset)}"
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("\n" + "-" * 70)
    print("BUILDING MODEL")
    print("-" * 70)

    model = FoveaNet().to(device)

    criterion = SmoothL1Loss()

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    # ========================================================
    # TRAINING VARIABLES
    # ========================================================

    best_val_error = float("inf")

    best_epoch = 0

    patience_counter = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_loss_total = 0.0

        train_error_total = 0.0

        train_samples = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]",
            leave=False
        )

        for images, targets in train_bar:

            images = images.to(
                device
            )

            targets = targets.to(
                device
            )

            # -----------------------------------------------
            # Forward
            # -----------------------------------------------

            optimizer.zero_grad()

            predictions = model(
                images
            )

            # -----------------------------------------------
            # Loss
            # -----------------------------------------------

            loss = criterion(
                predictions,
                targets
            )

            # -----------------------------------------------
            # Backpropagation
            # -----------------------------------------------

            loss.backward()

            optimizer.step()

            # -----------------------------------------------
            # Statistics
            # -----------------------------------------------

            batch_size = images.size(0)

            train_loss_total += (
                loss.item()
                * batch_size
            )

            errors = calculate_pixel_errors(
                predictions,
                targets
            )

            train_error_total += (
                np.sum(errors)
            )

            train_samples += batch_size

            # -----------------------------------------------
            # Progress bar
            # -----------------------------------------------

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = (
            train_loss_total
            / train_samples
        )

        train_error = (
            train_error_total
            / train_samples
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss_total = 0.0

        val_error_total = 0.0

        val_samples = 0

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch:02d}/{EPOCHS} [Val]",
            leave=False
        )

        with torch.no_grad():

            for images, targets in val_bar:

                images = images.to(
                    device
                )

                targets = targets.to(
                    device
                )

                predictions = model(
                    images
                )

                loss = criterion(
                    predictions,
                    targets
                )

                batch_size = images.size(0)

                val_loss_total += (
                    loss.item()
                    * batch_size
                )

                errors = calculate_pixel_errors(
                    predictions,
                    targets
                )

                val_error_total += (
                    np.sum(errors)
                )

                val_samples += batch_size

                val_bar.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

        val_loss = (
            val_loss_total
            / val_samples
        )

        val_error = (
            val_error_total
            / val_samples
        )

        # ====================================================
        # LEARNING RATE
        # ====================================================

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        # ====================================================
        # EPOCH RESULTS
        # ====================================================

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"Learning Rate     : {current_lr:.7f}"
        )

        print(
            f"Train Loss        : {train_loss:.5f}"
        )

        print(
            f"Train Error       : {train_error:.2f} px"
        )

        print(
            f"Val Loss          : {val_loss:.5f}"
        )

        print(
            f"Val Error         : {val_error:.2f} px"
        )

        print(
            f"Best Val Error    : "
            f"{min(best_val_error, val_error):.2f} px"
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if val_error < best_val_error:

            best_val_error = val_error

            best_epoch = epoch

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "val_error":
                        best_val_error,

                    "epoch":
                        best_epoch
                },
                SAVE_PATH
            )

            print(
                "✓ BEST MODEL SAVED"
            )

        else:

            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/{PATIENCE})"
            )

        print("-" * 70)

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if patience_counter >= PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            break

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("                 FOVEA TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best Epoch       : {best_epoch}"
    )

    print(
        f"Best Val Error   : "
        f"{best_val_error:.2f} pixels"
    )

    print(
        f"Checkpoint       : {SAVE_PATH}"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()