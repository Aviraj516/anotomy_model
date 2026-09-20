import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset


IMG_SIZE = 512


class FoveaDataset(Dataset):

    def __init__(self, data_root):

        self.data_root = data_root

        # ====================================================
        # IMAGE DIRECTORY
        # ====================================================

        self.image_dir = os.path.join(
            data_root,
            "A. Segmentation",
            "1. Original Images",
            "a. Training Set"
        )

        # ====================================================
        # FOVEA CSV
        # ====================================================

        self.csv_path = os.path.join(
            data_root,
            "C. Localization",
            "2. Groundtruths",
            "2. Fovea Center Location",
            "IDRiD_Fovea_Center_Training Set_Markups.csv"
        )

        print("\nImage directory:")
        print(self.image_dir)

        print("\nFovea CSV:")
        print(self.csv_path)

        if not os.path.exists(self.image_dir):

            raise FileNotFoundError(
                f"Image directory not found:\n{self.image_dir}"
            )

        if not os.path.exists(self.csv_path):

            raise FileNotFoundError(
                f"Fovea CSV not found:\n{self.csv_path}"
            )

        # ====================================================
        # READ CSV
        # ====================================================

        self.df = pd.read_csv(self.csv_path)

        print("\nCSV columns:")
        print(self.df.columns.tolist())

        # ====================================================
        # BUILD SAMPLES
        # ====================================================

        self.samples = self._build_samples()

        print(
            f"\nFovea samples found: {len(self.samples)}"
        )

        if len(self.samples) == 0:

            raise RuntimeError(
                "No valid Fovea samples found."
            )

    # ========================================================
    # BUILD SAMPLES
    # ========================================================

    def _build_samples(self):

        columns = [
            str(c).strip()
            for c in self.df.columns
        ]

        # ----------------------------------------------------
        # ACTUAL IDRiD CSV COLUMNS
        # ----------------------------------------------------

        image_col = "Image No"
        x_col = "X- Coordinate"
        y_col = "Y - Coordinate"

        # Safety check
        if image_col not in columns:

            raise ValueError(
                f"Could not find '{image_col}'"
            )

        if x_col not in columns:

            raise ValueError(
                f"Could not find '{x_col}'"
            )

        if y_col not in columns:

            raise ValueError(
                f"Could not find '{y_col}'"
            )

        print("\nDetected columns:")
        print("Image:", image_col)
        print("X    :", x_col)
        print("Y    :", y_col)

        # ----------------------------------------------------
        # CREATE IMAGE LOOKUP
        # ----------------------------------------------------

        image_files = {}

        for filename in os.listdir(
            self.image_dir
        ):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                stem = os.path.splitext(
                    filename
                )[0]

                image_files[stem] = os.path.join(
                    self.image_dir,
                    filename
                )

        print(
            f"\nImages available: "
            f"{len(image_files)}"
        )

        # ----------------------------------------------------
        # CREATE SAMPLES
        # ----------------------------------------------------

        samples = []

        for _, row in self.df.iterrows():

            image_name = str(
                row[image_col]
            ).strip()

            # Handle NaN
            if image_name.lower() == "nan":
                continue

            # Remove extension if present
            image_stem = os.path.splitext(
                os.path.basename(image_name)
            )[0]

            # ------------------------------------------------
            # FIX IDRiD NAMING
            #
            # CSV:
            # IDRiD_001
            #
            # Image:
            # IDRiD_01.jpg
            # ------------------------------------------------

            image_path = image_files.get(
                image_stem
            )

            if image_path is None:

                # Extract numeric portion
                try:

                    number = int(
                        image_stem.split("_")[-1]
                    )

                    # IDRiD_001 → IDRiD_01
                    possible_stem = (
                        f"IDRiD_{number:02d}"
                    )

                    image_path = image_files.get(
                        possible_stem
                    )

                except Exception:

                    image_path = None

            if image_path is None:

                continue

            # ------------------------------------------------
            # READ X/Y
            # ------------------------------------------------

            try:

                x = float(
                    row[x_col]
                )

                y = float(
                    row[y_col]
                )

            except Exception:

                continue

            # Reject invalid coordinates
            if x <= 0 or y <= 0:
                continue

            samples.append(
                (
                    image_path,
                    x,
                    y
                )
            )

        return samples

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.samples)

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, idx):

        image_path, x, y = self.samples[idx]

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise RuntimeError(
                f"Could not read image:\n{image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # ORIGINAL SIZE
        # ----------------------------------------------------

        original_h, original_w = image.shape[:2]

        # ----------------------------------------------------
        # NORMALIZE FOVEA COORDINATES
        # ----------------------------------------------------

        x_norm = x / original_w
        y_norm = y / original_h

        # Keep values in [0,1]
        x_norm = max(
            0.0,
            min(1.0, x_norm)
        )

        y_norm = max(
            0.0,
            min(1.0, y_norm)
        )

        # ----------------------------------------------------
        # RESIZE IMAGE
        # ----------------------------------------------------

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        image = image.astype(
            "float32"
        ) / 255.0

        # ----------------------------------------------------
        # HWC → CHW
        # ----------------------------------------------------

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(
            2,
            0,
            1
        )

        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        target = torch.tensor(
            [x_norm, y_norm],
            dtype=torch.float32
        )

        return image, target


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    PROJECT_ROOT = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../.."
        )
    )

    DATA_ROOT = os.path.join(
        PROJECT_ROOT,
        "data",
        "anatomy",
        "IDRid"
    )

    dataset = FoveaDataset(
        DATA_ROOT
    )

    print("\n" + "=" * 60)
    print("FOVEA DATASET TEST")
    print("=" * 60)

    print(
        "Total samples:",
        len(dataset)
    )

    image, target = dataset[0]

    print(
        "Image shape:",
        image.shape
    )

    print(
        "Image range:",
        image.min().item(),
        "to",
        image.max().item()
    )

    print(
        "Fovea target:",
        target
    )

    print("\nDataset test completed.")