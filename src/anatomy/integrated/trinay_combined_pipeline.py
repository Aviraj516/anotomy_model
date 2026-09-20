import os
import sys
import cv2
import numpy as np
import torch


# ============================================================
# TRINAY
# COMBINED ANATOMY + LESION PIPELINE
# ============================================================


# ============================================================
# PROJECT PATH SETUP
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Current directory:
#
# anotomy_model/
#   src/
#     anatomy/
#       integrated/   <-- CURRENT_DIR
#
# Therefore:
#
# integrated -> anatomy -> src -> anotomy_model
#

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "..",
        "..",
        ".."
    )
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)


# ------------------------------------------------------------
# Add project root
# ------------------------------------------------------------

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ------------------------------------------------------------
# Add src
# ------------------------------------------------------------

if SRC_DIR not in sys.path:

    sys.path.insert(
        0,
        SRC_DIR
    )


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device(
        "mps"
    )

elif torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


# ============================================================
# IMPORT LESION MODEL
# ============================================================

from anatomy.integrated.model import UNet


# ============================================================
# IMPORT LESION POST-PROCESSING
# ============================================================

from anatomy.integrated.postprocess import (
    postprocess_masks,
    summarize_masks
)


# ============================================================
# IMPORT LESION OVERLAY
# ============================================================

from anatomy.integrated.overlay import (
    create_overlay,
    create_comparison
)


# ============================================================
# IMPORT SPATIAL INTEGRATION
# ============================================================

from anatomy.integrated.spatial_integration import (
    build_spatial_analysis,
    add_lesion_spatial_analysis
)


# ============================================================
# IMPORT ANATOMY MODELS
# ============================================================

from anatomy.models.unet_efficientnet import (
    VesselUNet
)

from anatomy.models.optic_disc_unet import (
    OpticDiscUNet
)

from anatomy.models.fovea_net import (
    FoveaNet
)


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_SIZE = (
    512,
    512
)


# ============================================================
# LESION CONFIGURATION
# ============================================================

NUM_LESION_CLASSES = 4


LESION_NAMES = {
    0: "MA",
    1: "HE",
    2: "EX",
    3: "SE"
}


LESION_FULL_NAMES = {
    0: "Microaneurysm",
    1: "Haemorrhage",
    2: "Hard Exudate",
    3: "Soft Exudate"
}


# ============================================================
# EXISTING LESION THRESHOLDS
# ============================================================
#
# These are kept unchanged from your existing lesion model.
#
# MA = 0.30
# HE = 0.10
# EX = 0.25
# SE = 0.10
#
# ============================================================

LESION_THRESHOLDS = {
    0: 0.30,
    1: 0.10,
    2: 0.25,
    3: 0.10
}


# ============================================================
# ANATOMY THRESHOLDS
# ============================================================

VESSEL_THRESHOLD = 0.50

OPTIC_DISC_THRESHOLD = 0.50


# ============================================================
# MODEL CHECKPOINTS
# ============================================================

VESSEL_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "anatomy",
    "vessels",
    "best_vessel_unet.pth"
)


OPTIC_DISC_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "anatomy",
    "optic_disc",
    "best_optic_disc_unet.pth"
)


FOVEA_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "anatomy",
    "fovea",
    "best_fovea_net.pth"
)


LESION_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "anatomy",
    "lesions",
    "standard_unet_ddr_finetuned.pth"
)


# ============================================================
# CHECKPOINT CHECK
# ============================================================

def check_checkpoint(
    path,
    model_name
):

    if not os.path.exists(path):

        raise FileNotFoundError(
            "\n"
            + "=" * 70
            + "\nCHECKPOINT NOT FOUND\n"
            + "=" * 70
            + f"\nModel : {model_name}"
            + f"\nPath  : {path}"
            + "\n"
        )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_original_image(
    image_path
):

    # --------------------------------------------------------
    # Convert relative path to absolute
    # --------------------------------------------------------

    if not os.path.isabs(
        image_path
    ):

        image_path = os.path.join(
            PROJECT_ROOT,
            image_path
        )

    image_path = os.path.abspath(
        image_path
    )

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"\nImage not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise ValueError(
            f"\nOpenCV could not read image:\n"
            f"{image_path}"
        )

    return (
        image,
        image_path
    )


# ============================================================
# RETINA CROP
# ============================================================

def crop_retina(
    image
):

    """
    Same retinal crop concept used by the established
    lesion pipeline.
    """

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Retina mask
    # --------------------------------------------------------

    mask = (
        (gray > 10)
        .astype(
            np.uint8
        )
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
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if not contours:

        return image

    # --------------------------------------------------------
    # Largest contour
    # --------------------------------------------------------

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
    # Padding
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
    # Pad image if necessary
    # --------------------------------------------------------

    if any(
        [
            pad_left,
            pad_top,
            pad_right,
            pad_bottom
        ]
    ):

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
# PREPARE CANONICAL IMAGE
# ============================================================

def prepare_canonical_image(
    image
):

    # --------------------------------------------------------
    # Crop
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

    return resized


# ============================================================
# IMAGE TO TENSOR
# ============================================================

def image_to_tensor(
    image
):

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # NumPy -> Torch
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        rgb
    ).float() / 255.0

    # --------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------

    tensor = tensor.permute(
        2,
        0,
        1
    )

    # --------------------------------------------------------
    # Add batch
    # --------------------------------------------------------

    tensor = tensor.unsqueeze(
        0
    )

    return tensor


# ============================================================
# LOAD ANATOMY MODELS
# ============================================================

def load_anatomy_models():

    print(
        "\n"
        + "-" * 70
    )

    print(
        "LOADING ANATOMY MODELS"
    )

    print(
        "-" * 70
    )

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    check_checkpoint(
        VESSEL_MODEL_PATH,
        "Blood Vessel"
    )

    check_checkpoint(
        OPTIC_DISC_MODEL_PATH,
        "Optic Disc"
    )

    check_checkpoint(
        FOVEA_MODEL_PATH,
        "Fovea"
    )

    # --------------------------------------------------------
    # Create models
    # --------------------------------------------------------

    # The trained anatomy model classes already define their
    # architecture internally, so instantiate them without
    # constructor arguments. The checkpoint provides the
    # learned weights.
    vessel_model = VesselUNet()
    optic_disc_model = OpticDiscUNet()
    fovea_model = FoveaNet()

    # --------------------------------------------------------
    # Load checkpoints
    # --------------------------------------------------------

    vessel_checkpoint = torch.load(
        VESSEL_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    optic_checkpoint = torch.load(
        OPTIC_DISC_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    fovea_checkpoint = torch.load(
        FOVEA_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # --------------------------------------------------------
    # Support both raw state_dict and checkpoint dict
    # --------------------------------------------------------

    if (
        isinstance(
            vessel_checkpoint,
            dict
        )
        and "model_state_dict"
        in vessel_checkpoint
    ):

        vessel_state = (
            vessel_checkpoint[
                "model_state_dict"
            ]
        )

    else:

        vessel_state = (
            vessel_checkpoint
        )

    if (
        isinstance(
            optic_checkpoint,
            dict
        )
        and "model_state_dict"
        in optic_checkpoint
    ):

        optic_state = (
            optic_checkpoint[
                "model_state_dict"
            ]
        )

    else:

        optic_state = (
            optic_checkpoint
        )

    if (
        isinstance(
            fovea_checkpoint,
            dict
        )
        and "model_state_dict"
        in fovea_checkpoint
    ):

        fovea_state = (
            fovea_checkpoint[
                "model_state_dict"
            ]
        )

    else:

        fovea_state = (
            fovea_checkpoint
        )

    # --------------------------------------------------------
    # Load state dictionaries
    # --------------------------------------------------------

    vessel_model.load_state_dict(
        vessel_state
    )

    optic_disc_model.load_state_dict(
        optic_state
    )

    fovea_model.load_state_dict(
        fovea_state
    )

    # --------------------------------------------------------
    # Move to device
    # --------------------------------------------------------

    vessel_model.to(
        DEVICE
    )

    optic_disc_model.to(
        DEVICE
    )

    fovea_model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    vessel_model.eval()
    optic_disc_model.eval()
    fovea_model.eval()

    print(
        "✓ Vessel model loaded"
    )

    print(
        "✓ Optic disc model loaded"
    )

    print(
        "✓ Fovea model loaded"
    )

    return (
        vessel_model,
        optic_disc_model,
        fovea_model
    )


# ============================================================
# LOAD LESION MODEL
# ============================================================

def load_lesion_model():

    print(
        "\n"
        + "-" * 70
    )

    print(
        "LOADING LESION MODEL"
    )

    print(
        "-" * 70
    )

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    check_checkpoint(
        LESION_MODEL_PATH,
        "Lesion Standard U-Net"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=NUM_LESION_CLASSES
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        LESION_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # --------------------------------------------------------
    # Validate checkpoint
    # --------------------------------------------------------

    if (
        not isinstance(
            checkpoint,
            dict
        )
        or "model_state_dict"
        not in checkpoint
    ):

        raise KeyError(
            "\nExpected lesion checkpoint to contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    model.to(
        DEVICE
    )

    model.eval()

    print(
        "✓ Lesion Standard U-Net loaded"
    )

    print(
        f"  Epoch    : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"  Val Dice : "
        f"{checkpoint.get('val_dice', 'unknown')}"
    )

    return model


# ============================================================
# PREDICT ANATOMY
# ============================================================

def predict_anatomy(
    tensor,
    vessel_model,
    optic_disc_model,
    fovea_model
):

    print(
        "\nRunning anatomy models..."
    )

    with torch.no_grad():

        # ----------------------------------------------------
        # Vessel
        # ----------------------------------------------------

        vessel_output = vessel_model(
            tensor
        )

        # ----------------------------------------------------
        # Optic disc
        # ----------------------------------------------------

        optic_disc_output = optic_disc_model(
            tensor
        )

        # ----------------------------------------------------
        # Fovea
        # ----------------------------------------------------

        fovea_output = fovea_model(
            tensor
        )

    # ========================================================
    # VESSEL
    # ========================================================

    vessel_probability = torch.sigmoid(
        vessel_output
    )

    vessel_mask = (
        vessel_probability
        >= VESSEL_THRESHOLD
    )

    vessel_mask = (
        vessel_mask
        .squeeze()
        .cpu()
        .numpy()
        .astype(
            np.uint8
        )
    )

    # ========================================================
    # OPTIC DISC
    # ========================================================

    optic_disc_probability = torch.sigmoid(
        optic_disc_output
    )

    optic_disc_mask = (
        optic_disc_probability
        >= OPTIC_DISC_THRESHOLD
    )

    optic_disc_mask = (
        optic_disc_mask
        .squeeze()
        .cpu()
        .numpy()
        .astype(
            np.uint8
        )
    )

    # ========================================================
    # FOVEA
    # ========================================================

    fovea = (
        fovea_output
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # Fovea model outputs:
    #
    # [x_normalized, y_normalized]
    #

    fovea_x = float(
        fovea[0]
        * IMAGE_SIZE[0]
    )

    fovea_y = float(
        fovea[1]
        * IMAGE_SIZE[1]
    )

    # --------------------------------------------------------
    # Clip coordinates
    # --------------------------------------------------------

    fovea_x = float(
        np.clip(
            fovea_x,
            0,
            IMAGE_SIZE[0] - 1
        )
    )

    fovea_y = float(
        np.clip(
            fovea_y,
            0,
            IMAGE_SIZE[1] - 1
        )
    )

    print(
        "✓ Vessel prediction complete"
    )

    print(
        "✓ Optic disc prediction complete"
    )

    print(
        "✓ Fovea prediction complete"
    )

    return (
        vessel_mask,
        optic_disc_mask,
        fovea_x,
        fovea_y
    )


# ============================================================
# PREDICT LESIONS
# ============================================================

def predict_lesions(
    tensor,
    lesion_model
):

    print(
        "\nRunning lesion model..."
    )

    with torch.no_grad():

        logits = lesion_model(
            tensor
        )

        probabilities = torch.sigmoid(
            logits
        )

    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    probabilities = (
        probabilities
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Threshold each lesion
    # --------------------------------------------------------

    raw_masks = {}

    for class_idx in range(
        NUM_LESION_CLASSES
    ):

        threshold = (
            LESION_THRESHOLDS[
                class_idx
            ]
        )

        raw_masks[
            class_idx
        ] = (
            probabilities[
                class_idx
            ]
            >= threshold
        ).astype(
            np.uint8
        )

    print(
        "✓ Lesion prediction complete"
    )

    return (
        probabilities,
        raw_masks
    )


# ============================================================
# SINGLE LESION SPATIAL ANALYSIS
# ============================================================

def analyze_single_lesion_spatial(
    lesion_mask,
    macula_mask
):

    lesion_mask = (
        lesion_mask > 0
    ).astype(
        np.uint8
    )

    macula_mask = (
        macula_mask > 0
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Check shape
    # --------------------------------------------------------

    if (
        lesion_mask.shape
        != macula_mask.shape
    ):

        raise ValueError(
            "Lesion mask and macula mask "
            "must have the same shape."
        )

    # --------------------------------------------------------
    # Total lesion pixels
    # --------------------------------------------------------

    total_lesion_pixels = int(
        np.sum(
            lesion_mask
        )
    )

    # --------------------------------------------------------
    # No lesion
    # --------------------------------------------------------

    if total_lesion_pixels == 0:

        return {
            "total_lesion_pixels": 0,
            "macula_lesion_pixels": 0,
            "macula_overlap_ratio": 0.0
        }

    # --------------------------------------------------------
    # Lesion pixels in macula
    # --------------------------------------------------------

    macula_lesion_pixels = int(
        np.sum(
            lesion_mask[
                macula_mask > 0
            ]
        )
    )

    # --------------------------------------------------------
    # Ratio
    # --------------------------------------------------------

    overlap_ratio = (
        macula_lesion_pixels
        / total_lesion_pixels
    )

    return {
        "total_lesion_pixels":
            total_lesion_pixels,

        "macula_lesion_pixels":
            macula_lesion_pixels,

        "macula_overlap_ratio":
            float(
                overlap_ratio
            )
    }


# ============================================================
# SAVE MASK
# ============================================================

def save_binary_mask(
    mask,
    path
):

    mask_uint8 = (
        mask > 0
    ).astype(
        np.uint8
    ) * 255

    cv2.imwrite(
        path,
        mask_uint8
    )


# ============================================================
# SAVE ALL MASKS
# ============================================================

def save_all_masks(
    vessel_mask,
    optic_disc_mask,
    processed_lesion_masks,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Vessel
    # --------------------------------------------------------

    save_binary_mask(
        vessel_mask,
        os.path.join(
            output_dir,
            "vessel_mask.png"
        )
    )

    # --------------------------------------------------------
    # Optic disc
    # --------------------------------------------------------

    save_binary_mask(
        optic_disc_mask,
        os.path.join(
            output_dir,
            "optic_disc_mask.png"
        )
    )

    # --------------------------------------------------------
    # Lesions
    # --------------------------------------------------------

    for (
        class_idx,
        mask
    ) in processed_lesion_masks.items():

        name = LESION_NAMES[
            class_idx
        ]

        save_binary_mask(
            mask,
            os.path.join(
                output_dir,
                f"{name}_mask.png"
            )
        )


# ============================================================
# PRINT LESION SUMMARY
# ============================================================

def print_lesion_summary(
    summary
):

    print(
        "\n"
        + "-" * 70
    )

    print(
        "LESION SUMMARY"
    )

    print(
        "-" * 70
    )

    for (
        name,
        info
    ) in summary.items():

        if info["detected"]:

            status = "DETECTED"

        else:

            status = "NOT DETECTED"

        print(
            f"{name:<5}"
            f"{status:<18}"
            f"regions={info['regions']:<5}"
            f"pixels={info['pixels']}"
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_combined_pipeline(
    image_path
):

    print("\n")

    print(
        "=" * 78
    )

    print(
        "                         TRINAY"
    )

    print(
        "            COMBINED ANATOMY + LESION"
    )

    print(
        "                    PIPELINE"
    )

    print(
        "=" * 78
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    # ========================================================
    # STEP 1
    # ========================================================

    print(
        "\n"
        + "-" * 78
    )

    print(
        "STEP 1 — FUNDUS IMAGE"
    )

    print(
        "-" * 78
    )

    (
        original_image,
        absolute_image_path
    ) = load_original_image(
        image_path
    )

    print(
        f"\nImage:"
        f"\n  {absolute_image_path}"
    )

    print(
        "\n✓ Image loaded"
    )

    print(
        f"Original size:"
        f" {original_image.shape[1]}"
        f" x "
        f"{original_image.shape[0]}"
    )

    # ========================================================
    # STEP 2
    # ========================================================

    print(
        "\n"
        + "-" * 78
    )

    print(
        "STEP 2 — CANONICAL RETINAL IMAGE"
    )

    print(
        "-" * 78
    )

    canonical_image = (
        prepare_canonical_image(
            original_image
        )
    )

    tensor = (
        image_to_tensor(
            canonical_image
        )
        .to(
            DEVICE
        )
    )

    print(
        f"\n✓ Canonical image:"
        f" {canonical_image.shape[1]}"
        f" x "
        f"{canonical_image.shape[0]}"
    )

    # ========================================================
    # STEP 3
    # ========================================================

    print(
        "\n"
        + "-" * 78
    )

    print(
        "STEP 3 — LOAD MODELS"
    )

    print(
        "-" * 78
    )

    (
        vessel_model,
        optic_disc_model,
        fovea_model
    ) = load_anatomy_models()

    lesion_model = (
        load_lesion_model()
    )

    # ========================================================
    # STEP 4
    # ========================================================

    print(
        "\n"
        + "-" * 78
    )

    print(
        "STEP 4 — ANATOMY PREDICTION"
    )

    print(
        "-" * 78
    )

    (
        vessel_mask,
        optic_disc_mask,
        fovea_x,
        fovea_y
    ) = predict_anatomy(
        tensor,
        vessel_model,
        optic_disc_model,
        fovea_model
    )

    print(
        f"\nFovea:"
        f" ({fovea_x:.2f}, "
        f"{fovea_y:.2f})"
    )

    print(
        f"Vessel pixels:"
        f" {int(vessel_mask.sum())}"
    )

    print(
        f"Optic disc pixels:"
        f" {int(optic_disc_mask.sum())}"
    )

    # ========================================================
    # STEP 5
    # ========================================================

    print(
        "\n"
        + "-" * 78
    )

    print(
        "STEP 5 — LESION PREDICTION"
    )

    print(
        "-" * 78
    )

    (
        lesion_probabilities,
        raw_lesion_masks
    ) = predict_lesions(
        tensor,
        lesion_model
    )

    # ========================================================
    # STEP 6
    # ========================================================

    print(
        "\n"
        + "-" * 78
    )

    print(
        "STEP 6 — LESION POST-PROCESSING"
    )

    print(
        "-" * 78
    )

    processed_lesion_masks = (
        postprocess_masks(
            raw_lesion_masks,
            num_classes=NUM_LESION_CLASSES
        )
    )

    lesion_summary = (
        summarize_masks(
            processed_lesion_masks
        )
    )

    print(
        "✓ Post-processing complete"
    )

    print_lesion_summary(
        lesion_summary
    )

    # ========================================================
    # STEP 7
    # ========================================================

    print(
        "\n"
        + "-" * 78
    )

    print(
        "STEP 7 — SPATIAL INTEGRATION"
    )

    print(
        "-" * 78
    )

    spatial_results = (
        build_spatial_analysis(
            vessel_mask=vessel_mask,
            optic_disc_mask=optic_disc_mask,
            fovea_x=fovea_x,
            fovea_y=fovea_y
        )
    )

    print(
        "✓ Spatial anatomy analysis complete"
    )

    # ========================================================
    # STEP 8
    # ========================================================

    print(
        "\n"
        + "-" * 78
    )

    print(
        "STEP 8 — LESION ↔ MACULA ANALYSIS"
    )

    print(
        "-" * 78
    )

    macula_mask = (
        spatial_results[
            "macula_mask"
        ]
    )

    lesion_spatial_results = {}

    for class_idx in range(
        NUM_LESION_CLASSES
    ):

        name = LESION_NAMES[
            class_idx
        ]

        result = (
            analyze_single_lesion_spatial(
                processed_lesion_masks[
                    class_idx
                ],
                macula_mask
            )
        )

        lesion_spatial_results[
            name
        ] = result

    spatial_results[
        "lesion_spatial_analysis"
    ] = lesion_spatial_results

    # ========================================================
    # STEP 9
    # ========================================================

    print(
        "\n"
        + "-" * 78
    )

    print(
        "STEP 9 — SPATIAL RESULTS"
    )

    print(
        "-" * 78
    )

    vessel_analysis = (
        spatial_results[
            "vessel_analysis"
        ]
    )

    disc_fovea = (
        spatial_results[
            "disc_fovea"
        ]
    )

    print(
        f"Global vessel density : "
        f"{vessel_analysis['vessel_density']:.4f}"
    )

    print(
        f"Macular vessel density: "
        f"{vessel_analysis['macula_vessel_density']:.4f}"
        if vessel_analysis[
            "macula_vessel_density"
        ] is not None
        else
        "Macular vessel density: unavailable"
    )

    print(
        f"Connected components  : "
        f"{vessel_analysis['connected_components']}"
    )

    print(
        f"Largest component     : "
        f"{vessel_analysis['largest_component']}"
    )

    if disc_fovea is not None:

        print(
            f"Disc-Fovea distance   : "
            f"{disc_fovea['distance_pixels']:.2f} px"
        )

    else:

        print(
            "Disc-Fovea distance   : unavailable"
        )

    print(
        "\nLesion-macula relationship:"
    )

    for (
        name,
        result
    ) in lesion_spatial_results.items():

        print(
            f"{name:<5}"
            f"total={result['total_lesion_pixels']:<7}"
            f"macula={result['macula_lesion_pixels']:<7}"
            f"overlap={result['macula_overlap_ratio']:.4f}"
        )

    # ========================================================
    # STEP 10
    # ========================================================

    print(
        "\n"
        + "-" * 78
    )

    print(
        "STEP 10 — CREATE LESION OVERLAY"
    )

    print(
        "-" * 78
    )

    lesion_overlay = (
        create_overlay(
            canonical_image,
            processed_lesion_masks,
            lesion_summary
        )
    )

    lesion_comparison = (
        create_comparison(
            canonical_image,
            lesion_overlay
        )
    )

    print(
        "✓ Lesion overlay generated"
    )

    # ========================================================
    # STEP 11
    # ========================================================

    print(
        "\n"
        + "-" * 78
    )

    print(
        "STEP 11 — SAVE RESULTS"
    )

    print(
        "-" * 78
    )

    output_dir = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "trinay_combined"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Keep outputs separate when multiple fundus images are tested.
    image_stem = os.path.splitext(
        os.path.basename(absolute_image_path)
    )[0]

    # --------------------------------------------------------
    # Canonical image
    # --------------------------------------------------------

    canonical_path = os.path.join(
        output_dir,
        f"{image_stem}_canonical_retina.png"
    )

    cv2.imwrite(
        canonical_path,
        canonical_image
    )

    # --------------------------------------------------------
    # Lesion overlay
    # --------------------------------------------------------

    lesion_overlay_path = os.path.join(
        output_dir,
        f"{image_stem}_TRINAY_lesion_overlay.png"
    )

    cv2.imwrite(
        lesion_overlay_path,
        lesion_overlay
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison_path = os.path.join(
        output_dir,
        f"{image_stem}_TRINAY_lesion_comparison.png"
    )

    cv2.imwrite(
        comparison_path,
        lesion_comparison
    )

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    masks_dir = os.path.join(
        output_dir,
        "masks"
    )

    save_all_masks(
        vessel_mask,
        optic_disc_mask,
        processed_lesion_masks,
        masks_dir
    )

    print(
        "\n✓ Results saved"
    )

    # ========================================================
    # STEP 12
    # ========================================================

    print(
        "\n"
        + "-" * 78
    )

    print(
        "STEP 12 — CLEANUP"
    )

    print(
        "-" * 78
    )

    del tensor

    del vessel_model
    del optic_disc_model
    del fovea_model
    del lesion_model

    if DEVICE.type == "mps":

        try:

            torch.mps.empty_cache()

        except Exception:

            pass

    print(
        "✓ Memory cleanup complete"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "        TRINAY COMBINED PIPELINE COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        "\nOutput directory:"
    )

    print(
        f"  {output_dir}"
    )

    print(
        "\nCanonical retina:"
    )

    print(
        f"  {canonical_path}"
    )

    print(
        "\nLesion overlay:"
    )

    print(
        f"  {lesion_overlay_path}"
    )

    print(
        "\nLesion comparison:"
    )

    print(
        f"  {comparison_path}"
    )

    print(
        "\nMasks:"
    )

    print(
        f"  {masks_dir}"
    )

    print(
        "\n"
        + "=" * 78
    )

    return {
        "original_image":
            original_image,

        "canonical_image":
            canonical_image,

        "vessel_mask":
            vessel_mask,

        "optic_disc_mask":
            optic_disc_mask,

        "fovea":
            (
                fovea_x,
                fovea_y
            ),

        "lesion_probabilities":
            lesion_probabilities,

        "raw_lesion_masks":
            raw_lesion_masks,

        "processed_lesion_masks":
            processed_lesion_masks,

        "lesion_summary":
            lesion_summary,

        "spatial_results":
            spatial_results
    }


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            'python src/anatomy/integrated/'
            'trinay_combined_pipeline.py '
            '"path/to/image.jpg"'
        )

        print(
            "\nExample:"
        )

        print(
            'python src/anatomy/integrated/'
            'trinay_combined_pipeline.py '
            '"data/anatomy/IDRid/A. Segmentation/'
            '1. Original Images/a. Training Set/IDRiD_01.jpg"'
        )

        sys.exit(
            1
        )

    run_combined_pipeline(
        sys.argv[1]
    )