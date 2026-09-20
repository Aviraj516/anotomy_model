import os
import sys
import cv2
import numpy as np
import torch


# ============================================================
# TRINAY - PROJECT PATH SETUP
# ============================================================

# Current file:
# src/trinay_pipeline.py
#
# Project root:
# anotomy_model/
#
# We add the project root to Python's import path so that
# imports such as "src.overlay" work correctly.

CURRENT_FILE = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# TRINAY IMPORTS
# ============================================================

from model import UNet
from postprocess import (
    postprocess_masks,
    summarize_masks
)
from overlay import (
    create_overlay,
    create_comparison
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# ------------------------------------------------------------
# Established lesion model
# ------------------------------------------------------------

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "standard_unet_ddr_finetuned.pth"
)

IMAGE_SIZE = (
    512,
    512
)

NUM_CLASSES = 4


# ============================================================
# FROZEN LESION THRESHOLDS
# ============================================================

THRESHOLDS = {
    0: 0.30,   # MA - Microaneurysm
    1: 0.10,   # HE - Hemorrhage
    2: 0.25,   # EX - Hard Exudate
    3: 0.10    # SE - Soft Exudate
}


# ============================================================
# LESION CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "MA",
    1: "HE",
    2: "EX",
    3: "SE"
}


# ============================================================
# RETINA CROP
# ============================================================

def crop_retina(image):
    """
    Detect the non-black retinal region and create
    the same square crop used by the established
    lesion pipeline.

    Steps:
        1. Detect non-black retina
        2. Morphological closing
        3. Morphological opening
        4. Find largest contour
        5. Create square crop
        6. Add 5% margin
        7. Pad instead of clipping
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Detect retinal field
    # --------------------------------------------------------

    mask = (
        (gray > 10)
        .astype(np.uint8)
        * 255
    )

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    kernel = np.ones(
        (15, 15),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # --------------------------------------------------------
    # Find retinal contour
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    largest = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        largest
    )

    # --------------------------------------------------------
    # Square crop with 5% margin
    # --------------------------------------------------------

    side = int(
        max(w, h) * 1.05
    )

    cx = x + w // 2
    cy = y + h // 2

    x1 = cx - side // 2
    y1 = cy - side // 2

    x2 = x1 + side
    y2 = y1 + side

    img_h, img_w = image.shape[:2]

    # --------------------------------------------------------
    # Calculate required padding
    # --------------------------------------------------------

    pad_left = max(
        0,
        -x1
    )

    pad_top = max(
        0,
        -y1
    )

    pad_right = max(
        0,
        x2 - img_w
    )

    pad_bottom = max(
        0,
        y2 - img_h
    )

    # --------------------------------------------------------
    # Pad if crop exceeds image boundaries
    # --------------------------------------------------------

    if any([
        pad_left,
        pad_top,
        pad_right,
        pad_bottom
    ]):

        padded = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        x1 += pad_left
        x2 += pad_left

        y1 += pad_top
        y2 += pad_top

        crop = padded[
            y1:y2,
            x1:x2
        ]

    else:

        crop = image[
            y1:y2,
            x1:x2
        ]

    return crop


# ============================================================
# LOAD LESION MODEL
# ============================================================

def load_model():

    print("\nLoading segmentation model...")

    # --------------------------------------------------------
    # Verify checkpoint
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "\nLesion model checkpoint not found.\n\n"
            f"Expected location:\n{MODEL_PATH}\n\n"
            "Please check that standard_unet_ddr_finetuned.pth "
            "exists in the outputs folder."
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=NUM_CLASSES
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    model.to(DEVICE)
    model.eval()

    print("✓ Model loaded")

    print(
        f"Checkpoint epoch : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Validation Dice  : "
        f"{checkpoint.get('val_dice', 'unknown')}"
    )

    print(
        f"Device            : "
        f"{DEVICE}"
    )

    return model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def prepare_image(image):

    # --------------------------------------------------------
    # Retina crop
    # --------------------------------------------------------

    cropped = crop_retina(
        image
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized = cv2.resize(
        cropped,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        rgb
    ).float() / 255.0

    # HWC -> CHW

    tensor = tensor.permute(
        2,
        0,
        1
    )

    # Add batch dimension

    tensor = tensor.unsqueeze(
        0
    )

    return (
        resized,
        tensor
    )


# ============================================================
# LESION PREDICTION
# ============================================================

def predict(
    model,
    image
):

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    processed_image, tensor = prepare_image(
        image
    )

    tensor = tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            tensor
        )

        probabilities = torch.sigmoid(
            output
        )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    probabilities = (
        probabilities
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Apply frozen thresholds
    # --------------------------------------------------------

    masks = {}

    for class_idx in range(
        NUM_CLASSES
    ):

        masks[class_idx] = (
            probabilities[class_idx]
            >= THRESHOLDS[class_idx]
        ).astype(
            np.uint8
        )

    return (
        processed_image,
        probabilities,
        masks
    )


# ============================================================
# SAVE RAW MASKS
# ============================================================

def save_masks(
    masks,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    for idx in range(
        NUM_CLASSES
    ):

        name = CLASS_NAMES[idx]

        path = os.path.join(
            output_dir,
            f"{name}_raw_mask.png"
        )

        mask = masks[
            idx
        ]

        if mask.max() <= 1:
            mask = mask * 255

        cv2.imwrite(
            path,
            mask.astype(
                np.uint8
            )
        )


# ============================================================
# SAVE PROCESSED MASKS
# ============================================================

def save_processed_masks(
    masks,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    for idx in range(
        NUM_CLASSES
    ):

        name = CLASS_NAMES[idx]

        path = os.path.join(
            output_dir,
            f"{name}_processed_mask.png"
        )

        cv2.imwrite(
            path,
            masks[idx]
        )


# ============================================================
# PRINT LESION SUMMARY
# ============================================================

def print_lesion_summary(summary):

    print("\nLesion summary:")
    print("-" * 60)

    for name, info in summary.items():

        status = (
            "DETECTED"
            if info["detected"]
            else "NOT DETECTED"
        )

        print(
            f"  {name:<4} "
            f"{status:<15} "
            f"regions={info['regions']:<4} "
            f"pixels={info['pixels']}"
        )


# ============================================================
# MAIN TRINAY PIPELINE
# ============================================================

def run_trinay(
    image_path
):

    print("\n")
    print("=" * 78)
    print("                    TRINAY")
    print("              RETINAL ANALYSIS")
    print("=" * 78)

    # ========================================================
    # STEP 1 — INPUT IMAGE
    # ========================================================

    print("\n" + "-" * 78)
    print("STEP 1 — FUNDUS IMAGE INPUT")
    print("-" * 78)

    # --------------------------------------------------------
    # Resolve image path
    # --------------------------------------------------------

    if not os.path.isabs(image_path):

        image_path = os.path.join(
            PROJECT_ROOT,
            image_path
        )

    image_path = os.path.abspath(
        image_path
    )

    print(
        f"\nInput image:"
        f"\n  {image_path}"
    )

    print(
        f"\nDevice:"
        f"\n  {DEVICE}"
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise FileNotFoundError(
            f"\nCould not load image:\n{image_path}"
        )

    print(
        "\n✓ Image loaded"
    )

    print(
        f"Original size: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    # ========================================================
    # STEP 2 — LOAD LESION MODEL
    # ========================================================

    print("\n" + "-" * 78)
    print("STEP 2 — LESION SEGMENTATION")
    print("-" * 78)

    model = load_model()

    (
        processed_image,
        probabilities,
        raw_masks
    ) = predict(
        model,
        image
    )

    print(
        "\n✓ Lesion probabilities generated"
    )

    # ========================================================
    # STEP 3 — VISUAL POST-PROCESSING
    # ========================================================

    print("\n" + "-" * 78)
    print("STEP 3 — VISUAL POST-PROCESSING")
    print("-" * 78)

    processed_masks = postprocess_masks(
        raw_masks,
        num_classes=NUM_CLASSES
    )

    print(
        "\n✓ Visual post-processing completed"
    )

    # ========================================================
    # STEP 4 — LESION SUMMARY
    # ========================================================

    summary = summarize_masks(
        processed_masks
    )

    print_lesion_summary(
        summary
    )

    # ========================================================
    # STEP 5 — CLINICAL OVERLAY
    # ========================================================

    print("\n" + "-" * 78)
    print("STEP 4 — CLINICAL OVERLAY")
    print("-" * 78)

    overlay = create_overlay(
        processed_image,
        processed_masks,
        summary
    )

    comparison = create_comparison(
        processed_image,
        overlay
    )

    print(
        "\n✓ Non-color clinical overlay created"
    )

    # ========================================================
    # STEP 6 — SAVE OUTPUTS
    # ========================================================

    print("\n" + "-" * 78)
    print("STEP 5 — SAVE OUTPUTS")
    print("-" * 78)

    output_dir = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "trinay_result"
    )

    masks_dir = os.path.join(
        output_dir,
        "masks"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    original_output = os.path.join(
        output_dir,
        "processed_retina.png"
    )

    overlay_output = os.path.join(
        output_dir,
        "TRINAY_predicted_overlay.png"
    )

    comparison_output = os.path.join(
        output_dir,
        "TRINAY_comparison.png"
    )

    raw_masks_dir = os.path.join(
        masks_dir,
        "raw"
    )

    clean_masks_dir = os.path.join(
        masks_dir,
        "processed"
    )

    # --------------------------------------------------------
    # Save images
    # --------------------------------------------------------

    cv2.imwrite(
        original_output,
        processed_image
    )

    cv2.imwrite(
        overlay_output,
        overlay
    )

    cv2.imwrite(
        comparison_output,
        comparison
    )

    # --------------------------------------------------------
    # Save masks
    # --------------------------------------------------------

    save_masks(
        raw_masks,
        raw_masks_dir
    )

    save_processed_masks(
        processed_masks,
        clean_masks_dir
    )

    print(
        "\n✓ All outputs saved"
    )

    # ========================================================
    # DONE
    # ========================================================

    print("\n" + "=" * 78)
    print("                 TRINAY COMPLETED")
    print("=" * 78)

    print("\nOutputs:")

    print(
        f"\nProcessed retina:"
        f"\n  {original_output}"
    )

    print(
        f"\nClinical overlay:"
        f"\n  {overlay_output}"
    )

    print(
        f"\nComparison:"
        f"\n  {comparison_output}"
    )

    print(
        f"\nRaw masks:"
        f"\n  {raw_masks_dir}"
    )

    print(
        f"\nProcessed masks:"
        f"\n  {clean_masks_dir}"
    )

    print("\n" + "=" * 78)


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            'python -m src.trinay_pipeline '
            '"path/to/fundus_image.jpg"'
        )

        print(
            "\nExample:"
        )

        print(
            'python -m src.trinay_pipeline '
            '"data/anatomy/IDRid/A. Segmentation/1. Original Images/a. Training Set/IDRiD_01.jpg"'
        )

        sys.exit(1)

    run_trinay(
        sys.argv[1]
    )