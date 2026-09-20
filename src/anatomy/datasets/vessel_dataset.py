import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class VesselDataset(Dataset):

    def __init__(self, samples, image_size=(512, 512)):
        self.samples = samples
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        image_path, mask_path = self.samples[idx]

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask
        mask = np.array(Image.open(mask_path).convert("L"))

        # Resize
        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_AREA
        )

        mask = cv2.resize(
            mask,
            self.image_size,
            interpolation=cv2.INTER_NEAREST
        )

        # Normalize image
        image = image.astype(np.float32) / 255.0

        # Convert mask to binary
        mask = (mask > 0).astype(np.float32)

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        # Add channel to mask
        mask = np.expand_dims(mask, axis=0)

        return (
            image.astype(np.float32),
            mask.astype(np.float32)
        )


def get_drive_samples(root):

    samples = []

    train_images = os.path.join(
        root,
        "training",
        "images"
    )

    train_masks = os.path.join(
        root,
        "training",
        "1st_manual"
    )

    for filename in sorted(os.listdir(train_images)):

        if not filename.endswith(".tif"):
            continue

        image_path = os.path.join(
            train_images,
            filename
        )

        number = filename.split("_")[0]

        mask_path = os.path.join(
            train_masks,
            f"{number}_manual1.gif"
        )

        if os.path.exists(mask_path):
            samples.append(
                (image_path, mask_path)
            )

    return samples


def get_chase_samples(root):

    samples = []

    image_dir = os.path.join(
        root,
        "CHASE_DB1",
        "Images"
    )

    mask_dir = os.path.join(
        root,
        "CHASE_DB1",
        "Masks"
    )

    for filename in sorted(os.listdir(image_dir)):

        if not filename.lower().endswith(".jpg"):
            continue

        image_path = os.path.join(
            image_dir,
            filename
        )

        base = os.path.splitext(filename)[0]

        # Find corresponding mask
        possible_masks = [
            f"{base}.png",
            f"{base}_1stHO.png",
            f"{base}_1stHO.png"
        ]

        mask_path = None

        for mask_name in possible_masks:

            candidate = os.path.join(
                mask_dir,
                mask_name
            )

            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is not None:
            samples.append(
                (image_path, mask_path)
            )

    return samples


def get_hrf_samples(root):

    samples = []

    image_dir = os.path.join(
        root,
        "images"
    )

    mask_dir = os.path.join(
        root,
        "manual1"
    )

    for filename in sorted(os.listdir(image_dir)):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".tif", ".tiff", ".png")
        ):
            continue

        image_path = os.path.join(
            image_dir,
            filename
        )

        base = os.path.splitext(filename)[0]

        mask_path = os.path.join(
            mask_dir,
            f"{base}.tif"
        )

        if os.path.exists(mask_path):
            samples.append(
                (image_path, mask_path)
            )

    return samples


def get_all_vessel_samples(data_root):

    all_samples = []

    # DRIVE
    drive_path = os.path.join(
        data_root,
        "DRIVE"
    )

    if os.path.exists(drive_path):

        samples = get_drive_samples(drive_path)

        print(f"DRIVE samples: {len(samples)}")

        all_samples.extend(samples)

    # CHASE_DB1
    chase_path = os.path.join(
        data_root,
        "CHASE_db1"
    )

    if os.path.exists(chase_path):

        samples = get_chase_samples(chase_path)

        print(f"CHASE_DB1 samples: {len(samples)}")

        all_samples.extend(samples)

    # HRF
    hrf_path = os.path.join(
        data_root,
        "HRF"
    )

    if os.path.exists(hrf_path):

        samples = get_hrf_samples(hrf_path)

        print(f"HRF samples: {len(samples)}")

        all_samples.extend(samples)

    print(f"\nTOTAL vessel samples: {len(all_samples)}")

    return all_samples