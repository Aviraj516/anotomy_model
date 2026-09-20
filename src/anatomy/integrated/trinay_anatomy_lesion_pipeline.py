
"""
TRINAY
SEPARATE ANATOMY + LESION INFERENCE PIPELINE

Purpose:
    Prediction + visualization only.

Branches:
    Anatomy:
        - Blood vessels
        - Optic disc
        - Fovea
    Lesion:
        - MA
        - HE
        - EX
        - SE

This file intentionally does NOT run Integrated Gradients.
Keep XAI in trinay_xai_pipeline.py.

Run:
python src/anatomy/integrated/trinay_anatomy_lesion_pipeline.py \
"data/anatomy/IDRid/A. Segmentation/1. Original Images/a. Training Set/IDRiD_01.jpg"
"""

import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..", "..")
)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

from anatomy.integrated.model import UNet
from anatomy.integrated.postprocess import postprocess_masks, summarize_masks
from anatomy.models.unet_efficientnet import VesselUNet
from anatomy.models.optic_disc_unet import OpticDiscUNet
from anatomy.models.fovea_net import FoveaNet

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

IMAGE_SIZE = (512, 512)

LESION_NAMES = {
    0: "MA",
    1: "HE",
    2: "EX",
    3: "SE",
}

LESION_FULL_NAMES = {
    0: "Microaneurysm",
    1: "Haemorrhage",
    2: "Hard Exudate",
    3: "Soft Exudate",
}

LESION_THRESHOLDS = {
    0: 0.30,
    1: 0.10,
    2: 0.25,
    3: 0.10,
}

VESSEL_THRESHOLD = 0.50
OPTIC_DISC_THRESHOLD = 0.50

VESSEL_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "anatomy", "vessels",
    "best_vessel_unet.pth"
)

OPTIC_DISC_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "anatomy", "optic_disc",
    "best_optic_disc_unet.pth"
)

FOVEA_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "anatomy", "fovea",
    "best_fovea_net.pth"
)

LESION_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "anatomy", "lesions",
    "standard_unet_ddr_finetuned.pth"
)

# ------------------------------------------------------------
# UTILS
# ------------------------------------------------------------

def check_checkpoint(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n{name} checkpoint not found:\n{path}"
        )


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_original_image(image_path):
    if not os.path.isabs(image_path):
        image_path = os.path.join(PROJECT_ROOT, image_path)

    image_path = os.path.abspath(image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"\nImage not found:\n{image_path}")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"\nOpenCV could not read:\n{image_path}")

    return image, image_path


def crop_retina(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = ((gray > 10).astype(np.uint8)) * 255

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    side = max(1, int(max(w, h) * 1.05))

    cx = x + w // 2
    cy = y + h // 2

    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side

    img_h, img_w = image.shape[:2]

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_w)
    pad_bottom = max(0, y2 - img_h)

    if any((pad_left, pad_top, pad_right, pad_bottom)):
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

        return padded[y1:y2, x1:x2]

    return image[y1:y2, x1:x2]


def prepare_input(image):
    cropped = crop_retina(image)

    canonical = cv2.resize(
        cropped,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    rgb = cv2.cvtColor(canonical, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(
        rgb
    ).float() / 255.0

    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    return canonical, tensor.to(DEVICE)


# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------

def load_models():
    paths = [
        (VESSEL_MODEL_PATH, "Blood Vessel"),
        (OPTIC_DISC_MODEL_PATH, "Optic Disc"),
        (FOVEA_MODEL_PATH, "Fovea"),
        (LESION_MODEL_PATH, "Lesion"),
    ]

    for path, name in paths:
        check_checkpoint(path, name)

    vessel_model = VesselUNet()
    optic_disc_model = OpticDiscUNet()
    fovea_model = FoveaNet()
    lesion_model = UNet(
        in_channels=3,
        out_channels=4
    )

    checkpoints = [
        torch.load(
            VESSEL_MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        ),
        torch.load(
            OPTIC_DISC_MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        ),
        torch.load(
            FOVEA_MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        ),
        torch.load(
            LESION_MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        ),
    ]

    models = [
        vessel_model,
        optic_disc_model,
        fovea_model,
        lesion_model,
    ]

    names = [
        "Blood Vessel",
        "Optic Disc",
        "Fovea",
        "Lesion",
    ]

    for model, checkpoint, name in zip(
        models, checkpoints, names
    ):
        model.load_state_dict(
            extract_state_dict(checkpoint)
        )
        model.to(DEVICE)
        model.eval()
        print(f"✓ {name} model loaded")

    return tuple(models)


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

def predict_all(
    tensor,
    vessel_model,
    optic_disc_model,
    fovea_model,
    lesion_model
):
    with torch.no_grad():

        vessel_probability = torch.sigmoid(
            vessel_model(tensor)
        )[0, 0].cpu().numpy()

        optic_disc_probability = torch.sigmoid(
            optic_disc_model(tensor)
        )[0, 0].cpu().numpy()

        fovea_output = (
            fovea_model(tensor)
            .squeeze(0)
            .cpu()
            .numpy()
        )

        lesion_probability = torch.sigmoid(
            lesion_model(tensor)
        )[0].cpu().numpy()

    vessel_mask = (
        vessel_probability >= VESSEL_THRESHOLD
    ).astype(np.uint8)

    optic_disc_mask = (
        optic_disc_probability >= OPTIC_DISC_THRESHOLD
    ).astype(np.uint8)

    fovea_x = float(
        np.clip(
            fovea_output[0] * IMAGE_SIZE[0],
            0,
            IMAGE_SIZE[0] - 1
        )
    )

    fovea_y = float(
        np.clip(
            fovea_output[1] * IMAGE_SIZE[1],
            0,
            IMAGE_SIZE[1] - 1
        )
    )

    raw_lesion_masks = {}

    for class_idx in range(4):
        raw_lesion_masks[class_idx] = (
            lesion_probability[class_idx]
            >= LESION_THRESHOLDS[class_idx]
        ).astype(np.uint8)

    processed_lesion_masks = postprocess_masks(
        raw_lesion_masks,
        num_classes=4
    )

    lesion_summary = summarize_masks(
        processed_lesion_masks
    )

    return {
        "vessel_probability": vessel_probability,
        "vessel_mask": vessel_mask,
        "optic_disc_probability": optic_disc_probability,
        "optic_disc_mask": optic_disc_mask,
        "fovea_x": fovea_x,
        "fovea_y": fovea_y,
        "lesion_probability": lesion_probability,
        "raw_lesion_masks": raw_lesion_masks,
        "processed_lesion_masks": processed_lesion_masks,
        "lesion_summary": lesion_summary,
    }


# ------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------

def mask_overlay(
    image,
    mask,
    color,
    alpha=0.55
):
    output = image.copy()
    mask_bool = mask > 0

    if not np.any(mask_bool):
        return output

    color_layer = np.zeros_like(output)
    color_layer[:, :] = color

    output[mask_bool] = (
        output[mask_bool].astype(np.float32) * (1 - alpha)
        + color_layer[mask_bool].astype(np.float32) * alpha
    ).astype(np.uint8)

    return output


def create_anatomy_overlay(
    canonical,
    vessel_mask,
    optic_disc_mask,
    fovea_x,
    fovea_y
):
    output = canonical.copy()

    # Green vessels.
    output = mask_overlay(
        output,
        vessel_mask,
        (0, 255, 0),
        alpha=0.35
    )

    # Draw optic disc contour in blue.
    contours, _ = cv2.findContours(
        optic_disc_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        cv2.drawContours(
            output,
            contours,
            -1,
            (255, 0, 0),
            2
        )

    # Fovea marker in yellow.
    center = (
        int(round(fovea_x)),
        int(round(fovea_y))
    )

    cv2.circle(
        output,
        center,
        8,
        (0, 255, 255),
        -1
    )

    cv2.circle(
        output,
        center,
        15,
        (255, 255, 255),
        2
    )

    return output


def create_lesion_overlay(
    canonical,
    processed_lesion_masks
):
    # BGR colors.
    lesion_colors = {
        0: (0, 0, 255),       # MA red
        1: (180, 0, 180),     # HE violet
        2: (0, 220, 255),     # EX yellow
        3: (255, 190, 0),     # SE cyan-ish
    }

    output = canonical.copy()

    for class_idx, mask in processed_lesion_masks.items():
        output = mask_overlay(
            output,
            mask,
            lesion_colors[class_idx],
            alpha=0.55
        )

    return output


def create_combined_overlay(
    canonical,
    vessel_mask,
    optic_disc_mask,
    fovea_x,
    fovea_y,
    processed_lesion_masks
):
    output = create_anatomy_overlay(
        canonical,
        vessel_mask,
        optic_disc_mask,
        fovea_x,
        fovea_y
    )

    lesion_colors = {
        0: (0, 0, 255),
        1: (180, 0, 180),
        2: (0, 220, 255),
        3: (255, 190, 0),
    }

    for class_idx, mask in processed_lesion_masks.items():
        output = mask_overlay(
            output,
            mask,
            lesion_colors[class_idx],
            alpha=0.45
        )

    return output


def save_dashboard(
    original,
    anatomy_overlay,
    lesion_overlay,
    combined_overlay,
    result,
    path
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 12)
    )

    panels = [
        ("Original Fundus", original),
        ("Anatomy Model", anatomy_overlay),
        ("Lesion Model", lesion_overlay),
        ("Combined Anatomy + Lesions", combined_overlay),
    ]

    for ax, (title, image) in zip(
        axes.flat, panels
    ):
        ax.imshow(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )
        )
        ax.set_title(title)
        ax.axis("off")

    detected = []

    for class_idx, info in result["lesion_summary"].items():
        if not info.get("detected", False):
            continue

        if isinstance(class_idx, str):
            lesion_name = class_idx
        else:
            lesion_name = LESION_NAMES.get(
                int(class_idx),
                str(class_idx)
            )

        detected.append(lesion_name)

    fig.suptitle(
        "TRINAY - ANATOMY + LESION ANALYSIS",
        fontsize=18,
        fontweight="bold"
    )

    fig.text(
        0.5,
        0.02,
        "Fovea: "
        f"({result['fovea_x']:.1f}, {result['fovea_y']:.1f})"
        "    |    Lesions: "
        + (", ".join(detected) if detected else "None"),
        ha="center",
        fontsize=11
    )

    plt.tight_layout(rect=(0, 0.04, 1, 0.96))

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def run_pipeline(image_path):

    print("\n")
    print("=" * 78)
    print("                         TRINAY")
    print("             ANATOMY + LESION PIPELINE")
    print("=" * 78)
    print(f"\nDevice: {DEVICE}")

    # 1. Image
    print("\n" + "-" * 78)
    print("STEP 1 — FUNDUS IMAGE")
    print("-" * 78)

    original, absolute_path = load_original_image(image_path)

    print(f"✓ Image loaded")
    print(
        f"Original size: "
        f"{original.shape[1]} x {original.shape[0]}"
    )

    # 2. Canonical image
    print("\n" + "-" * 78)
    print("STEP 2 — CANONICAL RETINA")
    print("-" * 78)

    canonical, tensor = prepare_input(original)

    print("✓ Canonical retina: 512 x 512")

    # 3. Models
    print("\n" + "-" * 78)
    print("STEP 3 — LOAD MODELS")
    print("-" * 78)

    (
        vessel_model,
        optic_disc_model,
        fovea_model,
        lesion_model
    ) = load_models()

    # 4. Predictions
    print("\n" + "-" * 78)
    print("STEP 4 — PREDICT ANATOMY + LESIONS")
    print("-" * 78)

    result = predict_all(
        tensor,
        vessel_model,
        optic_disc_model,
        fovea_model,
        lesion_model
    )

    print("✓ Anatomy prediction complete")
    print(
        f"  Vessel pixels: "
        f"{int(result['vessel_mask'].sum())}"
    )
    print(
        f"  Optic disc pixels: "
        f"{int(result['optic_disc_mask'].sum())}"
    )
    print(
        f"  Fovea: "
        f"({result['fovea_x']:.2f}, "
        f"{result['fovea_y']:.2f})"
    )

    print("✓ Lesion prediction complete")

    for class_idx, info in result["lesion_summary"].items():
        if isinstance(class_idx, str):
            lesion_name = class_idx
        else:
            lesion_name = LESION_NAMES[int(class_idx)]

        print(
            f"  {lesion_name}: "
            f"{'DETECTED' if info.get('detected', False) else 'NOT DETECTED'}"
            f" | regions={info.get('regions', 0)}"
            f" | pixels={info.get('pixels', 0)}"
        )

    # 5. Visualizations
    print("\n" + "-" * 78)
    print("STEP 5 — CREATE VISUALIZATIONS")
    print("-" * 78)

    anatomy_overlay = create_anatomy_overlay(
        canonical,
        result["vessel_mask"],
        result["optic_disc_mask"],
        result["fovea_x"],
        result["fovea_y"]
    )

    lesion_overlay = create_lesion_overlay(
        canonical,
        result["processed_lesion_masks"]
    )

    combined_overlay = create_combined_overlay(
        canonical,
        result["vessel_mask"],
        result["optic_disc_mask"],
        result["fovea_x"],
        result["fovea_y"],
        result["processed_lesion_masks"]
    )

    # 6. Save
    filename = os.path.splitext(
        os.path.basename(absolute_path)
    )[0]

    output_dir = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "trinay_anatomy_lesion",
        filename
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    canonical_path = os.path.join(
        output_dir,
        "canonical_retina.png"
    )

    anatomy_path = os.path.join(
        output_dir,
        "anatomy_overlay.png"
    )

    lesion_path = os.path.join(
        output_dir,
        "lesion_overlay.png"
    )

    combined_path = os.path.join(
        output_dir,
        "combined_anatomy_lesion.png"
    )

    dashboard_path = os.path.join(
        output_dir,
        "TRINAY_ANATOMY_LESION_DASHBOARD.png"
    )

    cv2.imwrite(canonical_path, canonical)
    cv2.imwrite(anatomy_path, anatomy_overlay)
    cv2.imwrite(lesion_path, lesion_overlay)
    cv2.imwrite(combined_path, combined_overlay)

    save_dashboard(
        original,
        anatomy_overlay,
        lesion_overlay,
        combined_overlay,
        result,
        dashboard_path
    )

    print("✓ Anatomy visualization saved")
    print("✓ Lesion visualization saved")
    print("✓ Combined visualization saved")
    print("✓ Dashboard saved")

    print("\n" + "=" * 78)
    print("        ANATOMY + LESION PIPELINE COMPLETE")
    print("=" * 78)

    print(f"\nOutput directory:\n{output_dir}")
    print(f"\nMain dashboard:\n{dashboard_path}")

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

    return result


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "\nUsage:\n"
            "python src/anatomy/integrated/"
            "trinay_anatomy_lesion_pipeline.py "
            '"path/to/image.jpg"'
        )
        sys.exit(1)

    run_pipeline(sys.argv[1])
