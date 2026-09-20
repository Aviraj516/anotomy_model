
"""
TRINAY - FAST SEPARATE XAI PIPELINE

Purpose
-------
Generate fast, lightweight Integrated Gradients visualizations for:
    1. Anatomy models
       - Blood vessels
       - Optic disc
       - Fovea
    2. Lesion model
       - MA
       - HE
       - EX
       - SE
    3. Combined anatomy + lesion XAI

Important
---------
This script is separate from the main prediction pipeline.

It is deliberately optimized for a MacBook:
    - XAI runs on CPU, not MPS.
    - XAI input is 192x192.
    - Integrated Gradients uses only 2 steps.
    - Only detected lesion classes are explained.
    - At most 1200 target pixels are explained.
    - Models are loaded one at a time.
    - No 512x512 Captum tensor is kept on MPS.

The resulting heatmaps are resized to 512x512 only for visualization.

Run
---
python src/anatomy/integrated/trinay_xai_fast.py \
"data/anatomy/IDRid/A. Segmentation/1. Original Images/a. Training Set/IDRiD_01.jpg"

Dependency
----------
pip install captum
"""

import os
import sys
import gc
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients


# ============================================================
# PROJECT PATH
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

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

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# DEVICE
# ============================================================

# XAI is deliberately CPU-only.
# This prevents Captum backward passes from consuming the
# shared MPS memory pool and hanging the laptop.
XAI_DEVICE = torch.device("cpu")


# ============================================================
# IMPORT MODELS
# ============================================================

from anatomy.integrated.model import UNet
from anatomy.models.unet_efficientnet import VesselUNet
from anatomy.models.optic_disc_unet import OpticDiscUNet
from anatomy.models.fovea_net import FoveaNet


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = (512, 512)

# XAI computation size.
# 192x192 is intentionally small for speed.
XAI_SIZE = (192, 192)

# Very low IG steps for fast visualization.
IG_STEPS = 2

# Only strongest predicted regions are explained.
TARGET_PERCENTILE = 92

# Cap the target pixels.
MAX_TARGET_PIXELS = 1200

# Do not waste XAI time on absent lesions.
ONLY_DETECTED_LESIONS = True

# Heatmap normalization.
LOW_PERCENTILE = 75
HIGH_PERCENTILE = 99

# Existing lesion thresholds.
LESION_THRESHOLDS = {
    0: 0.30,   # MA
    1: 0.10,   # HE
    2: 0.25,   # EX
    3: 0.10,   # SE
}

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

VESSEL_THRESHOLD = 0.50
OPTIC_DISC_THRESHOLD = 0.50


# ============================================================
# CHECKPOINTS
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
# OUTPUT HELPERS
# ============================================================

def output_directory(image_path):
    filename = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    return os.path.join(
        PROJECT_ROOT,
        "outputs",
        "trinay_xai_fast",
        filename
    )


def check_file(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n{name} not found:\n{path}"
        )


def cleanup_cpu():
    gc.collect()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(image_path):
    if not os.path.isabs(image_path):
        image_path = os.path.join(
            PROJECT_ROOT,
            image_path
        )

    image_path = os.path.abspath(
        image_path
    )

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"\nImage not found:\n{image_path}"
        )

    image = cv2.imread(
        image_path
    )

    if image is None:
        raise ValueError(
            f"\nOpenCV could not read:\n{image_path}"
        )

    return image, image_path


def crop_retina(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    mask = (
        (gray > 10)
        .astype(np.uint8)
        * 255
    )

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

    side = max(
        1,
        int(max(w, h) * 1.05)
    )

    cx = x + w // 2
    cy = y + h // 2

    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side

    img_h, img_w = image.shape[:2]

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

        return padded[
            y1:y2,
            x1:x2
        ]

    return image[
        y1:y2,
        x1:x2
    ]


def prepare_xai_input(image):
    cropped = crop_retina(
        image
    )

    canonical = cv2.resize(
        cropped,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    small = cv2.resize(
        canonical,
        XAI_SIZE,
        interpolation=cv2.INTER_AREA
    )

    rgb = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2RGB
    )

    tensor = (
        torch.from_numpy(rgb)
        .float()
        .div(255.0)
        .permute(
            2,
            0,
            1
        )
        .unsqueeze(0)
        .to(XAI_DEVICE)
    )

    return canonical, tensor


# ============================================================
# MODEL FACTORIES
# ============================================================

def load_vessel_model():
    check_file(
        VESSEL_MODEL_PATH,
        "Vessel checkpoint"
    )

    model = VesselUNet()

    checkpoint = torch.load(
        VESSEL_MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    state = (
        checkpoint["model_state_dict"]
        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        )
        else checkpoint
    )

    model.load_state_dict(
        state
    )

    model.to(XAI_DEVICE)
    model.eval()

    return model


def load_optic_disc_model():
    check_file(
        OPTIC_DISC_MODEL_PATH,
        "Optic disc checkpoint"
    )

    model = OpticDiscUNet()

    checkpoint = torch.load(
        OPTIC_DISC_MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    state = (
        checkpoint["model_state_dict"]
        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        )
        else checkpoint
    )

    model.load_state_dict(
        state
    )

    model.to(XAI_DEVICE)
    model.eval()

    return model


def load_fovea_model():
    check_file(
        FOVEA_MODEL_PATH,
        "Fovea checkpoint"
    )

    model = FoveaNet()

    checkpoint = torch.load(
        FOVEA_MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    state = (
        checkpoint["model_state_dict"]
        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        )
        else checkpoint
    )

    model.load_state_dict(
        state
    )

    model.to(XAI_DEVICE)
    model.eval()

    return model


def load_lesion_model():
    check_file(
        LESION_MODEL_PATH,
        "Lesion checkpoint"
    )

    model = UNet(
        in_channels=3,
        out_channels=4
    )

    checkpoint = torch.load(
        LESION_MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    if (
        not isinstance(checkpoint, dict)
        or "model_state_dict" not in checkpoint
    ):
        raise KeyError(
            "Lesion checkpoint must contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(XAI_DEVICE)
    model.eval()

    return model


# ============================================================
# XAI CORE
# ============================================================

def normalize_map(value):
    value = np.asarray(
        value,
        dtype=np.float32
    )

    if value.size == 0:
        return np.zeros_like(
            value,
            dtype=np.float32
        )

    low = float(
        np.percentile(
            value,
            LOW_PERCENTILE
        )
    )

    high = float(
        np.percentile(
            value,
            HIGH_PERCENTILE
        )
    )

    if high <= low:
        low = float(
            value.min()
        )
        high = float(
            value.max()
        )

    if high - low < 1e-8:
        return np.zeros_like(
            value,
            dtype=np.float32
        )

    value = np.clip(
        value,
        low,
        high
    )

    value = (
        value - low
    ) / (
        high - low + 1e-8
    )

    return cv2.GaussianBlur(
        value.astype(np.float32),
        (0, 0),
        0.7
    )


def attribution_to_map(attribution):
    values = (
        attribution
        .detach()
        .abs()
        .sum(dim=1)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return normalize_map(
        values
    )


def build_target(probability, threshold):
    predicted = (
        probability >= threshold
    )

    if not torch.any(predicted):
        return None

    positive = probability[
        predicted
    ]

    level = torch.quantile(
        positive,
        TARGET_PERCENTILE / 100.0
    )

    target = (
        probability >= level
    ).float()

    count = int(
        target.sum().item()
    )

    if count <= MAX_TARGET_PIXELS:
        return target

    flat_probability = (
        probability.flatten()
    )

    flat_target = (
        target.flatten()
    )

    positions = torch.nonzero(
        flat_target > 0,
        as_tuple=False
    ).squeeze(1)

    values = (
        flat_probability[
            positions
        ]
    )

    keep = min(
        MAX_TARGET_PIXELS,
        positions.numel()
    )

    _, selected = torch.topk(
        values,
        keep
    )

    reduced = torch.zeros_like(
        flat_target
    )

    reduced[
        positions[selected]
    ] = 1.0

    return reduced.reshape(
        target.shape
    )


def integrated_gradients_segmentation(
    model,
    tensor,
    target,
    class_idx
):
    if target is None:
        return None

    target = target.to(
        XAI_DEVICE
    )

    def forward_fn(x):
        logits = model(x)

        output = logits[
            :,
            class_idx
        ]

        return (
            output * target
        ).sum(
            dim=(1, 2)
        )

    x = (
        tensor
        .detach()
        .clone()
        .requires_grad_(True)
    )

    baseline = torch.zeros_like(
        x
    )

    ig = IntegratedGradients(
        forward_fn
    )

    attribution = ig.attribute(
        x,
        baselines=baseline,
        n_steps=IG_STEPS,
        internal_batch_size=1,
        method="riemann_trapezoid"
    )

    result = attribution_to_map(
        attribution
    )

    del attribution
    del baseline
    del x
    del ig

    cleanup_cpu()

    return result


def integrated_gradients_regression(
    model,
    tensor,
    coordinate_idx
):
    def forward_fn(x):
        return model(x)[
            :,
            coordinate_idx
        ]

    x = (
        tensor
        .detach()
        .clone()
        .requires_grad_(True)
    )

    baseline = torch.zeros_like(
        x
    )

    ig = IntegratedGradients(
        forward_fn
    )

    attribution = ig.attribute(
        x,
        baselines=baseline,
        n_steps=IG_STEPS,
        internal_batch_size=1,
        method="riemann_trapezoid"
    )

    result = attribution_to_map(
        attribution
    )

    del attribution
    del baseline
    del x
    del ig

    cleanup_cpu()

    return result


# ============================================================
# PREDICT DETECTED LESIONS
# ============================================================

def get_lesion_predictions(
    lesion_model,
    tensor
):
    with torch.no_grad():
        logits = lesion_model(
            tensor
        )

        probabilities = torch.sigmoid(
            logits
        )[0]

    return probabilities.cpu()


def detected_lesion_classes(
    probabilities
):
    detected = []

    for class_idx in range(4):
        pixel_count = int(
            (
                probabilities[class_idx]
                >= LESION_THRESHOLDS[
                    class_idx
                ]
            )
            .sum()
            .item()
        )

        if pixel_count > 0:
            detected.append(
                class_idx
            )

    return detected


# ============================================================
# ANATOMY XAI
# ============================================================

def explain_vessels(
    model,
    tensor
):
    print(
        "  • Blood Vessel XAI..."
    )

    with torch.no_grad():
        probability = torch.sigmoid(
            model(tensor)
        )[0, 0]

    target = build_target(
        probability,
        VESSEL_THRESHOLD
    )

    result = integrated_gradients_segmentation(
        model,
        tensor,
        target,
        0
    )

    del probability
    del target

    return result


def explain_optic_disc(
    model,
    tensor
):
    print(
        "  • Optic Disc XAI..."
    )

    with torch.no_grad():
        probability = torch.sigmoid(
            model(tensor)
        )[0, 0]

    target = build_target(
        probability,
        OPTIC_DISC_THRESHOLD
    )

    result = integrated_gradients_segmentation(
        model,
        tensor,
        target,
        0
    )

    del probability
    del target

    return result


def explain_fovea(
    model,
    tensor
):
    print(
        "  • Fovea XAI..."
    )

    x_map = integrated_gradients_regression(
        model,
        tensor,
        0
    )

    y_map = integrated_gradients_regression(
        model,
        tensor,
        1
    )

    result = np.maximum(
        x_map,
        y_map
    )

    del x_map
    del y_map

    return normalize_map(
        result
    )


# ============================================================
# LESION XAI
# ============================================================

def explain_lesions(
    model,
    tensor,
    detected_classes
):
    results = {}

    for class_idx in detected_classes:

        print(
            f"  • {LESION_NAMES[class_idx]} XAI..."
        )

        with torch.no_grad():
            probabilities = torch.sigmoid(
                model(tensor)
            )[0, class_idx]

        target = build_target(
            probabilities,
            LESION_THRESHOLDS[
                class_idx
            ]
        )

        result = integrated_gradients_segmentation(
            model,
            tensor,
            target,
            class_idx
        )

        if result is not None:
            results[class_idx] = result

        del probabilities
        del target

    return results


# ============================================================
# VISUALIZATION
# ============================================================

def resize_map(xai_map):
    return cv2.resize(
        xai_map,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )


def create_overlay(
    image,
    xai_map,
    alpha=0.65
):
    """
    Keep the original fundus visible and add XAI only where
    attribution is meaningful.

    Low-attribution pixels remain almost completely unchanged,
    so the result does NOT become a solid blue heatmap.
    """

    xai_map = np.clip(
        np.asarray(xai_map, dtype=np.float32),
        0.0,
        1.0
    )

    # Suppress weak attribution so the retina remains visible.
    threshold = 0.15

    strength = np.clip(
        (xai_map - threshold) / (1.0 - threshold),
        0.0,
        1.0
    )

    # Make weak explanations fade smoothly.
    strength = np.power(
        strength,
        1.35
    )

    heat_uint8 = (
        xai_map * 255.0
    ).astype(
        np.uint8
    )

    # JET gives the same familiar red/yellow high-attribution
    # appearance, but its blue/green low values are mostly
    # hidden because their alpha is near zero.
    heatmap = cv2.applyColorMap(
        heat_uint8,
        cv2.COLORMAP_JET
    )

    # Slightly smooth only the alpha mask.
    strength = cv2.GaussianBlur(
        strength,
        (0, 0),
        0.8
    )

    effective_alpha = (
        strength * alpha
    )[:, :, None]

    image_float = image.astype(
        np.float32
    )

    heat_float = heatmap.astype(
        np.float32
    )

    blended = (
        image_float
        * (1.0 - effective_alpha)
        +
        heat_float
        * effective_alpha
    )

    return np.clip(
        blended,
        0,
        255
    ).astype(
        np.uint8
    )


def add_title(
    image,
    title
):
    canvas = np.full(
        (
            image.shape[0] + 52,
            image.shape[1],
            3
        ),
        245,
        dtype=np.uint8
    )

    canvas[52:] = image

    cv2.putText(
        canvas,
        "TRINAY XAI",
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (25, 25, 25),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        title,
        (180, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (80, 80, 80),
        1,
        cv2.LINE_AA
    )

    return canvas


def save_overlay(
    image,
    xai_map,
    title,
    path
):
    overlay = create_overlay(
        image,
        resize_map(
            xai_map
        )
    )

    overlay = add_title(
        overlay,
        title
    )

    cv2.imwrite(
        path,
        overlay
    )


# ============================================================
# DASHBOARD
# ============================================================

def save_dashboard(
    canonical,
    anatomy_maps,
    lesion_maps,
    output_dir
):
    anatomy_combined = np.maximum.reduce(
        [
            normalize_map(x)
            for x in anatomy_maps.values()
        ]
    )

    if lesion_maps:
        lesion_combined = np.maximum.reduce(
            [
                normalize_map(x)
                for x in lesion_maps.values()
            ]
        )
    else:
        lesion_combined = np.zeros(
            IMAGE_SIZE,
            dtype=np.float32
        )

    combined = normalize_map(
        np.maximum(
            anatomy_combined,
            lesion_combined
        )
    )

    panels = [
        (
            "Original",
            canonical
        ),
        (
            "Anatomy XAI",
            create_overlay(
                canonical,
                resize_map(
                    anatomy_combined
                )
            )
        ),
        (
            "Lesion XAI",
            create_overlay(
                canonical,
                resize_map(
                    lesion_combined
                )
            )
        ),
        (
            "Combined Anatomy + Lesion",
            create_overlay(
                canonical,
                resize_map(
                    combined
                ),
                alpha=0.60
            )
        )
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(13, 3.5)
    )

    for ax, (
        title,
        image
    ) in zip(
        axes,
        panels
    ):
        ax.imshow(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )
        )
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        "TRINAY - FAST XAI",
        fontsize=14,
        fontweight="bold"
    )

    fig.tight_layout()

    path = os.path.join(
        output_dir,
        "TRINAY_XAI_DASHBOARD.png"
    )

    fig.savefig(
        path,
        dpi=90,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# MAIN
# ============================================================

def run_xai(
    image_path
):
    print("\n")
    print("=" * 78)
    print("                 TRINAY FAST XAI")
    print("=" * 78)

    print(
        f"\nXAI device : {XAI_DEVICE}"
    )
    print(
        f"XAI size   : "
        f"{XAI_SIZE[0]} x {XAI_SIZE[1]}"
    )
    print(
        f"IG steps   : {IG_STEPS}"
    )
    print(
        "Only detected lesions: "
        f"{ONLY_DETECTED_LESIONS}"
    )

    # --------------------------------------------------------
    # 1. IMAGE
    # --------------------------------------------------------

    print(
        "\n" + "-" * 78
    )
    print(
        "STEP 1 — LOAD IMAGE"
    )
    print(
        "-" * 78
    )

    image, absolute_path = load_image(
        image_path
    )

    canonical, tensor = prepare_xai_input(
        image
    )

    print(
        f"✓ Original: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    print(
        "✓ Canonical: 512 x 512"
    )

    print(
        "✓ XAI input: "
        f"{XAI_SIZE[0]} x {XAI_SIZE[1]}"
    )

    # --------------------------------------------------------
    # 2. LESION PREDICTION
    # --------------------------------------------------------

    print(
        "\n" + "-" * 78
    )
    print(
        "STEP 2 — CHECK DETECTED LESIONS"
    )
    print(
        "-" * 78
    )

    lesion_model = load_lesion_model()

    probabilities = get_lesion_predictions(
        lesion_model,
        tensor
    )

    detected = detected_lesion_classes(
        probabilities
    )

    if detected:
        for class_idx in detected:
            pixel_count = int(
                (
                    probabilities[class_idx]
                    >= LESION_THRESHOLDS[
                        class_idx
                    ]
                )
                .sum()
                .item()
            )

            print(
                f"✓ {LESION_NAMES[class_idx]} "
                f"{LESION_FULL_NAMES[class_idx]} "
                f"pixels={pixel_count}"
            )
    else:
        print(
            "No lesion classes detected."
        )

    del probabilities
    del lesion_model
    cleanup_cpu()

    # --------------------------------------------------------
    # 3. ANATOMY XAI
    # --------------------------------------------------------

    print(
        "\n" + "-" * 78
    )
    print(
        "STEP 3 — ANATOMY XAI"
    )
    print(
        "-" * 78
    )

    vessel_model = load_vessel_model()

    vessel_map = explain_vessels(
        vessel_model,
        tensor
    )

    del vessel_model
    cleanup_cpu()

    optic_model = load_optic_disc_model()

    optic_map = explain_optic_disc(
        optic_model,
        tensor
    )

    del optic_model
    cleanup_cpu()

    fovea_model = load_fovea_model()

    fovea_map = explain_fovea(
        fovea_model,
        tensor
    )

    del fovea_model
    cleanup_cpu()

    anatomy_maps = {
        "vessel": vessel_map,
        "optic_disc": optic_map,
        "fovea": fovea_map
    }

    print(
        "✓ Anatomy XAI complete"
    )

    # --------------------------------------------------------
    # 4. LESION XAI
    # --------------------------------------------------------

    print(
        "\n" + "-" * 78
    )
    print(
        "STEP 4 — LESION XAI"
    )
    print(
        "-" * 78
    )

    # Reload only the lesion model when needed.
    # It is never kept together with all anatomy models.
    lesion_model = load_lesion_model()

    lesion_maps = explain_lesions(
        lesion_model,
        tensor,
        detected
    )

    del lesion_model
    cleanup_cpu()

    print(
        "✓ Lesion XAI complete"
    )

    # --------------------------------------------------------
    # 5. SAVE
    # --------------------------------------------------------

    print(
        "\n" + "-" * 78
    )
    print(
        "STEP 5 — SAVE XAI RESULTS"
    )
    print(
        "-" * 78
    )

    out_dir = output_directory(
        absolute_path
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    # Anatomy images.
    save_overlay(
        canonical,
        anatomy_maps["vessel"],
        "Integrated Gradients | Blood Vessels",
        os.path.join(
            out_dir,
            "ANATOMY_VESSEL_XAI.png"
        )
    )

    save_overlay(
        canonical,
        anatomy_maps["optic_disc"],
        "Integrated Gradients | Optic Disc",
        os.path.join(
            out_dir,
            "ANATOMY_OPTIC_DISC_XAI.png"
        )
    )

    save_overlay(
        canonical,
        anatomy_maps["fovea"],
        "Integrated Gradients | Fovea",
        os.path.join(
            out_dir,
            "ANATOMY_FOVEA_XAI.png"
        )
    )

    # Lesion images.
    for class_idx, xai_map in lesion_maps.items():
        save_overlay(
            canonical,
            xai_map,
            (
                "Integrated Gradients | "
                f"{LESION_FULL_NAMES[class_idx]}"
            ),
            os.path.join(
                out_dir,
                f"LESION_{LESION_NAMES[class_idx]}_XAI.png"
            )
        )

    # Combined maps.
    anatomy_combined = np.maximum.reduce(
        list(
            anatomy_maps.values()
        )
    )

    lesion_combined = (
        np.maximum.reduce(
            list(
                lesion_maps.values()
            )
        )
        if lesion_maps
        else np.zeros(
            IMAGE_SIZE,
            dtype=np.float32
        )
    )

    combined_xai = normalize_map(
        np.maximum(
            anatomy_combined,
            lesion_combined
        )
    )

    save_overlay(
        canonical,
        anatomy_combined,
        "Combined Anatomy XAI",
        os.path.join(
            out_dir,
            "ANATOMY_COMBINED_XAI.png"
        )
    )

    save_overlay(
        canonical,
        lesion_combined,
        "Combined Lesion XAI",
        os.path.join(
            out_dir,
            "LESION_COMBINED_XAI.png"
        )
    )

    save_overlay(
        canonical,
        combined_xai,
        "Combined Anatomy + Lesion XAI",
        os.path.join(
            out_dir,
            "TRINAY_COMBINED_XAI.png"
        )
    )

    dashboard = save_dashboard(
        canonical,
        anatomy_maps,
        lesion_maps,
        out_dir
    )

    print(
        "✓ All XAI images saved"
    )

    print(
        f"\nOutput directory:\n{out_dir}"
    )

    print(
        f"\nDashboard:\n{dashboard}"
    )

    # Final cleanup.
    del tensor
    del image
    del canonical
    del anatomy_maps
    del lesion_maps
    del combined_xai

    cleanup_cpu()

    print(
        "\n" + "=" * 78
    )
    print(
        "                 TRINAY FAST XAI COMPLETE"
    )
    print(
        "=" * 78
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "\nUsage:\n"
            "python src/anatomy/integrated/"
            "trinay_xai_fast.py "
            '"path/to/fundus_image.jpg"'
        )

        sys.exit(1)

    run_xai(
        sys.argv[1]
    )
