import os
import sys

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)

sys.path.append(PROJECT_ROOT)

from src.anatomy.models.unet_efficientnet import VesselUNet


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "anatomy",
    "vessels",
    "best_vessel_unet.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "anatomy",
    "vessels"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (512, 512)

THRESHOLD = 0.5


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("RETINAL VESSEL INFERENCE")
print("EfficientNet-B0 + U-Net")
print("=" * 70)

print("\nDevice:", DEVICE)

print("\nLoading model...")

model = VesselUNet()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

# ------------------------------------------------------------
# Handle checkpoint format
# ------------------------------------------------------------

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Validation Dice: "
        f"{checkpoint.get('val_dice', 'unknown')}"
    )

else:

    model.load_state_dict(checkpoint)


model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image:\n{image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original_image = image.copy()

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    image = image.astype(
        np.float32
    ) / 255.0

    # HWC → CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    image_tensor = torch.from_numpy(
        image
    ).float()

    return original_image, image_tensor


# ============================================================
# PREDICTION
# ============================================================

def predict_vessels(image_path):

    original_image, image_tensor = preprocess_image(
        image_path
    )

    image_tensor = image_tensor.to(DEVICE)

    print("\nRunning inference...")

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probability = torch.sigmoid(
            output
        )

    probability = probability.squeeze().cpu().numpy()

    # --------------------------------------------------------
    # Binary vessel mask
    # --------------------------------------------------------

    mask = (
        probability >= THRESHOLD
    ).astype(np.uint8) * 255

    # --------------------------------------------------------
    # Resize prediction to original image
    # --------------------------------------------------------

    original_h, original_w = original_image.shape[:2]

    mask_original = cv2.resize(
        mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )

    probability_original = cv2.resize(
        probability,
        (original_w, original_h),
        interpolation=cv2.INTER_LINEAR
    )

    return (
        original_image,
        probability_original,
        mask_original
    )


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(
    original_image,
    mask
):

    overlay = original_image.copy()

    vessel_pixels = mask > 0

    # Highlight vessels
    overlay[vessel_pixels] = [
        255,
        0,
        0
    ]

    blended = cv2.addWeighted(
        original_image,
        0.65,
        overlay,
        0.35,
        0
    )

    return blended


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    image_path,
    original_image,
    probability,
    mask,
    overlay
):

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    original_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_original.png"
    )

    probability_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_probability.png"
    )

    mask_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_vessel_mask.png"
    )

    overlay_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_vessel_overlay.png"
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    cv2.imwrite(
        original_path,
        cv2.cvtColor(
            original_image,
            cv2.COLOR_RGB2BGR
        )
    )

    # --------------------------------------------------------
    # Probability map
    # --------------------------------------------------------

    probability_uint8 = (
        probability * 255
    ).astype(np.uint8)

    cv2.imwrite(
        probability_path,
        probability_uint8
    )

    # --------------------------------------------------------
    # Binary mask
    # --------------------------------------------------------

    cv2.imwrite(
        mask_path,
        mask
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    cv2.imwrite(
        overlay_path,
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )

    print("\nResults saved:")

    print(
        "Original   :",
        original_path
    )

    print(
        "Probability:",
        probability_path
    )

    print(
        "Mask       :",
        mask_path
    )

    print(
        "Overlay    :",
        overlay_path
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    original_image,
    probability,
    mask,
    overlay,
    image_path
):

    plt.figure(
        figsize=(16, 5)
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    plt.subplot(
        1,
        4,
        1
    )

    plt.imshow(
        original_image
    )

    plt.title(
        "Original Fundus"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    plt.subplot(
        1,
        4,
        2
    )

    plt.imshow(
        probability,
        cmap="gray"
    )

    plt.title(
        "Vessel Probability"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # Binary mask
    # --------------------------------------------------------

    plt.subplot(
        1,
        4,
        3
    )

    plt.imshow(
        mask,
        cmap="gray"
    )

    plt.title(
        f"Vessel Mask\nThreshold={THRESHOLD}"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    plt.subplot(
        1,
        4,
        4
    )

    plt.imshow(
        overlay
    )

    plt.title(
        "Vessel Overlay"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")

        print(
            'python src/anatomy/inference/'
            'predict_vessels.py "IMAGE_PATH"'
        )

        print("\nExample:")

        print(
            'python src/anatomy/inference/'
            'predict_vessels.py '
            '"data/anatomy/DRIVE/training/images/21_training.tif"'
        )

        sys.exit(1)


    image_path = sys.argv[1]

    if not os.path.exists(image_path):

        print(
            f"\nERROR: Image not found:\n{image_path}"
        )

        sys.exit(1)


    print("\nInput image:")
    print(image_path)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    (
        original_image,
        probability,
        mask
    ) = predict_vessels(
        image_path
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = create_overlay(
        original_image,
        mask
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    vessel_pixels = np.sum(
        mask > 0
    )

    total_pixels = mask.size

    vessel_percentage = (
        vessel_pixels /
        total_pixels
    ) * 100

    print("\nPrediction statistics:")

    print(
        f"Vessel pixels: "
        f"{vessel_pixels:,}"
    )

    print(
        f"Total pixels : "
        f"{total_pixels:,}"
    )

    print(
        f"Vessel area  : "
        f"{vessel_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        image_path,
        original_image,
        probability,
        mask,
        overlay
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_results(
        original_image,
        probability,
        mask,
        overlay,
        image_path
    )

    print("\nInference completed successfully.")