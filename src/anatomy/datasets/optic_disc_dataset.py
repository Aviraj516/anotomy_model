import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class OpticDiscDataset(Dataset):

    def __init__(
        self,
        samples,
        image_size=(512, 512)
    ):

        self.samples = samples
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        image_path, mask_path = self.samples[idx]

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mask = np.array(
            Image.open(mask_path).convert("L")
        )

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

        image = image.astype(
            np.float32
        ) / 255.0

        mask = (
            mask > 0
        ).astype(
            np.float32
        )

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        mask = np.expand_dims(
            mask,
            axis=0
        )

        return (
            image.astype(np.float32),
            mask.astype(np.float32)
        )


def get_idrid_optic_disc_samples(data_root):

    samples = []

    idrid_root = os.path.join(
        data_root,
        "IDRid"
    )

    image_dir = os.path.join(
        idrid_root,
        "A. Segmentation",
        "1. Original Images",
        "a. Training Set"
    )

    mask_dir = os.path.join(
        idrid_root,
        "A. Segmentation",
        "2. All Segmentation Groundtruths",
        "a. Training Set",
        "5. Optic Disc"
    )

    if not os.path.exists(image_dir):

        raise FileNotFoundError(
            f"Image directory not found:\n{image_dir}"
        )

    if not os.path.exists(mask_dir):

        raise FileNotFoundError(
            f"Optic Disc mask directory not found:\n{mask_dir}"
        )

    for filename in sorted(
        os.listdir(image_dir)
    ):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(
            image_dir,
            filename
        )

        base = os.path.splitext(
            filename
        )[0]

        mask_path = os.path.join(
            mask_dir,
            f"{base}_OD.tif"
        )

        if os.path.exists(mask_path):

            samples.append(
                (
                    image_path,
                    mask_path
                )
            )

    print(
        f"IDRiD Optic Disc samples: "
        f"{len(samples)}"
    )

    return samples