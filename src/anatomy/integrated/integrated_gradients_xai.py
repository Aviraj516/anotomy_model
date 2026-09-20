import os
import sys
import gc

import cv2
import numpy as np
import torch

from captum.attr import IntegratedGradients

from anatomy.integrated.model import UNet


# ============================================================
# TRINAY - LIGHTWEIGHT INTEGRATED GRADIENTS XAI
# ============================================================

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

MODEL_PATH = (
    "outputs/standard_unet_ddr_finetuned.pth"
)

IMAGE_SIZE = (512, 512)

NUM_CLASSES = 4

CLASS_NAMES = {
    0: "MA",
    1: "HE",
    2: "EX",
    3: "SE"
}

CLASS_FULL_NAMES = {
    0: "Microaneurysm",
    1: "Haemorrhage",
    2: "Hard Exudate",
    3: "Soft Exudate"
}

# Frozen inference thresholds
THRESHOLDS = {
    0: 0.30,
    1: 0.10,
    2: 0.25,
    3: 0.10
}

# ------------------------------------------------------------
# SPEED SETTINGS
# ------------------------------------------------------------

# Lower = faster.
# 16 is a good presentation/demo compromise.
IG_STEPS = 16

# Only the strongest predicted pixels are used as
# the explanation target.
TARGET_PERCENTILE = 80

# Maximum number of target pixels.
# Prevents extremely large attribution calculations.
MAX_TARGET_PIXELS = 12000

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = (
    "outputs/trinay_result/XAI"
)


# ============================================================
# MEMORY CLEANUP
# ============================================================

def cleanup_memory():

    gc.collect()

    if torch.backends.mps.is_available():

        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# ============================================================
# RETINA CROP
# ============================================================

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

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_w)
    pad_bottom = max(0, y2 - img_h)

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


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(image):

    cropped = crop_retina(
        image
    )

    resized = cv2.resize(
        cropped,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    tensor = (
        torch.from_numpy(rgb)
        .float()
        / 255.0
    )

    tensor = tensor.permute(
        2,
        0,
        1
    )

    tensor = tensor.unsqueeze(0)

    return resized, tensor


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading TRINAY model...")

    model = UNet(
        in_channels=3,
        out_channels=NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    print("✓ Model loaded")

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown"
        )
    )

    return model


# ============================================================
# GET PREDICTIONS
# ============================================================

def get_predictions(
    model,
    input_tensor
):

    with torch.no_grad():

        logits = model(
            input_tensor
        )

        probabilities = torch.sigmoid(
            logits
        )

    return probabilities


# ============================================================
# BUILD SMALL EXPLANATION TARGET
# ============================================================

def build_target_mask(
    probability_map,
    threshold
):

    # --------------------------------------------------------
    # Start with normal predicted lesion
    # --------------------------------------------------------

    predicted = (
        probability_map >= threshold
    )

    if predicted.sum().item() == 0:

        # No lesion detected.
        # We don't want to run XAI for this class.
        return None

    # --------------------------------------------------------
    # Use stronger predicted pixels.
    # This reduces the amount of gradient computation.
    # --------------------------------------------------------

    positive_values = probability_map[
        predicted
    ]

    percentile_value = torch.quantile(
        positive_values,
        TARGET_PERCENTILE / 100.0
    )

    target = (
        probability_map
        >= percentile_value
    )

    # --------------------------------------------------------
    # Limit target size
    # --------------------------------------------------------

    indices = torch.nonzero(
        target,
        as_tuple=False
    )

    if len(indices) > MAX_TARGET_PIXELS:

        values = probability_map[
            target
        ]

        _, top_indices = torch.topk(
            values,
            MAX_TARGET_PIXELS
        )

        flat_target = torch.zeros_like(
            probability_map.flatten()
        )

        target_positions = torch.nonzero(
            target.flatten(),
            as_tuple=False
        ).squeeze(1)

        selected_positions = (
            target_positions[
                top_indices
            ]
        )

        flat_target[
            selected_positions
        ] = 1.0

        target = flat_target.reshape(
            probability_map.shape
        )

    return target.float()


# ============================================================
# FORWARD FUNCTION
# ============================================================

def make_forward_function(
    model,
    class_idx,
    target_mask
):

    def forward_func(x):

        logits = model(x)

        selected = (
            logits[:, class_idx]
            * target_mask
        )

        return selected.sum(
            dim=(1, 2)
        )

    return forward_func


# ============================================================
# NORMALIZE ATTRIBUTION
# ============================================================

def normalize_attribution(
    attribution
):

    # Combine RGB attribution
    attribution = (
        attribution
        .abs()
        .sum(dim=1)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    # Remove tiny numerical noise
    attribution[
        attribution < 0
    ] = 0

    if attribution.max() <= 0:

        return np.zeros(
            attribution.shape,
            dtype=np.uint8
        )

    # Normalize
    low = np.percentile(
        attribution,
        60
    )

    high = np.percentile(
        attribution,
        99
    )

    if high <= low:

        low = attribution.min()
        high = attribution.max()

    attribution = np.clip(
        attribution,
        low,
        high
    )

    attribution = (
        attribution - low
    ) / (
        high - low + 1e-8
    )

    attribution = (
        attribution * 255
    ).astype(
        np.uint8
    )

    # Smooth for visualization
    attribution = cv2.GaussianBlur(
        attribution,
        (0, 0),
        2
    )

    return attribution


# ============================================================
# CREATE HEATMAP
# ============================================================

def create_heatmap(
    attribution
):

    return cv2.applyColorMap(
        attribution,
        cv2.COLORMAP_JET
    )


# ============================================================
# HEATMAP OVERLAY
# ============================================================

def create_xai_overlay(
    image,
    heatmap,
    attribution
):

    weight = (
        attribution.astype(
            np.float32
        ) / 255.0
    )

    # Only show meaningful attribution
    weight = np.clip(
        weight,
        0.0,
        1.0
    )

    weight = cv2.GaussianBlur(
        weight,
        (0, 0),
        2
    )

    weight = np.expand_dims(
        weight,
        axis=2
    )

    image_float = image.astype(
        np.float32
    )

    heatmap_float = heatmap.astype(
        np.float32
    )

    alpha = 0.65

    result = (
        image_float
        * (
            1.0
            - alpha * weight
        )
        +
        heatmap_float
        * (
            alpha * weight
        )
    )

    result = np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )

    return result


# ============================================================
# LABEL
# ============================================================

def add_label(
    image,
    class_idx
):

    h, w = image.shape[:2]

    bar_height = 55

    canvas = np.full(
        (
            h + bar_height,
            w,
            3
        ),
        245,
        dtype=np.uint8
    )

    canvas[
        bar_height:
    ] = image

    cv2.putText(
        canvas,
        "TRINAY XAI",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (25, 25, 25),
        2,
        cv2.LINE_AA
    )

    text = (
        "Integrated Gradients | "
        + CLASS_FULL_NAMES[class_idx]
    )

    cv2.putText(
        canvas,
        text,
        (180, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (80, 80, 80),
        1,
        cv2.LINE_AA
    )

    return canvas


# ============================================================
# EXPLAIN ONE LESION
# ============================================================

def explain_class(
    model,
    input_tensor,
    processed_image,
    class_idx,
    probability_map
):

    threshold = THRESHOLDS[
        class_idx
    ]

    target_mask = build_target_mask(
        probability_map,
        threshold
    )

    # No prediction
    if target_mask is None:

        return None

    print(
        f"  Running Integrated Gradients "
        f"({IG_STEPS} steps)..."
    )

    forward_func = make_forward_function(
        model,
        class_idx,
        target_mask.unsqueeze(0)
    )

    ig = IntegratedGradients(
        forward_func
    )

    baseline = torch.zeros_like(
        input_tensor
    )

    # Important for Captum
    input_tensor = (
        input_tensor
        .clone()
        .detach()
        .requires_grad_(True)
    )

    # --------------------------------------------------------
    # Integrated Gradients
    # --------------------------------------------------------

    attribution = ig.attribute(
        input_tensor,
        baselines=baseline,
        n_steps=IG_STEPS,
        internal_batch_size=1
    )

    attribution_gray = (
        normalize_attribution(
            attribution
        )
    )

    heatmap = create_heatmap(
        attribution_gray
    )

    xai_overlay = create_xai_overlay(
        processed_image,
        heatmap,
        attribution_gray
    )

    xai_overlay = add_label(
        xai_overlay,
        class_idx
    )

    # Cleanup
    del attribution
    del attribution_gray
    del heatmap
    del target_mask
    del baseline
    del input_tensor
    del ig

    cleanup_memory()

    return xai_overlay


# ============================================================
# MAIN XAI
# ============================================================

def run_xai(
    image_path
):

    print("\n")
    print("=" * 78)
    print("       TRINAY — LIGHTWEIGHT INTEGRATED GRADIENTS XAI")
    print("=" * 78)

    print(
        f"\nDevice     : {DEVICE}"
    )

    print(
        f"IG steps   : {IG_STEPS}"
    )

    print(
        f"Input image: {image_path}"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load image:\n{image_path}"
        )

    processed_image, input_tensor = (
        prepare_image(image)
    )

    input_tensor = input_tensor.to(
        DEVICE
    )

    print(
        f"\nImage size : "
        f"{processed_image.shape[1]} x "
        f"{processed_image.shape[0]}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # ONE NORMAL FORWARD PASS
    # ========================================================

    print(
        "\nGenerating normal segmentation..."
    )

    probabilities = get_predictions(
        model,
        input_tensor
    )

    print(
        "✓ Segmentation generated"
    )

    # ========================================================
    # DETECT WHICH LESIONS EXIST
    # ========================================================

    detected_classes = []

    print(
        "\nDetected lesion classes:"
    )

    for class_idx in range(
        NUM_CLASSES
    ):

        probability_map = (
            probabilities[
                0,
                class_idx
            ]
        )

        threshold = THRESHOLDS[
            class_idx
        ]

        pixel_count = int(
            (
                probability_map
                >= threshold
            ).sum().item()
        )

        name = CLASS_NAMES[
            class_idx
        ]

        if pixel_count > 0:

            detected_classes.append(
                class_idx
            )

            print(
                f"  ✓ {name:<3} "
                f"{CLASS_FULL_NAMES[class_idx]:<22} "
                f"pixels={pixel_count}"
            )

        else:

            print(
                f"  - {name:<3} "
                f"not detected"
            )

    # ========================================================
    # NO LESIONS
    # ========================================================

    if not detected_classes:

        print(
            "\nNo lesions detected."
        )

        print(
            "No Integrated Gradients calculation needed."
        )

        del probabilities
        del input_tensor
        del model

        cleanup_memory()

        return

    # ========================================================
    # XAI PER DETECTED LESION
    # ========================================================

    generated = []

    for class_idx in detected_classes:

        name = CLASS_NAMES[
            class_idx
        ]

        print(
            "\n"
            + "-" * 65
        )

        print(
            f"XAI: {name} — "
            f"{CLASS_FULL_NAMES[class_idx]}"
        )

        probability_map = (
            probabilities[
                0,
                class_idx
            ]
        )

        try:

            result = explain_class(
                model,
                input_tensor,
                processed_image,
                class_idx,
                probability_map
            )

            if result is not None:

                output_path = os.path.join(
                    OUTPUT_DIR,
                    f"{name}_XAI_overlay.png"
                )

                cv2.imwrite(
                    output_path,
                    result
                )

                generated.append(
                    (
                        name,
                        result
                    )
                )

                print(
                    f"  ✓ Saved: {output_path}"
                )

        except Exception as e:

            print(
                f"  ✗ XAI failed for {name}"
            )

            print(
                f"    {type(e).__name__}: {e}"
            )

        cleanup_memory()

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    if generated:

        print(
            "\n"
            + "-" * 65
        )

        print(
            "Creating combined XAI image..."
        )

        tiles = [
            item[1]
            for item in generated
        ]

        # Same height
        max_h = max(
            tile.shape[0]
            for tile in tiles
        )

        max_w = max(
            tile.shape[1]
            for tile in tiles
        )

        normalized = []

        for tile in tiles:

            canvas = np.full(
                (
                    max_h,
                    max_w,
                    3
                ),
                245,
                dtype=np.uint8
            )

            h, w = tile.shape[:2]

            canvas[
                :h,
                :w
            ] = tile

            normalized.append(
                canvas
            )

        # One or two classes
        if len(normalized) == 1:

            combined = normalized[0]

        else:

            rows = []

            for i in range(
                0,
                len(normalized),
                2
            ):

                row = normalized[i]

                if i + 1 < len(normalized):

                    row = np.hstack([
                        row,
                        normalized[i + 1]
                    ])

                rows.append(
                    row
                )

            # Equal widths
            row_width = max(
                row.shape[1]
                for row in rows
            )

            fixed_rows = []

            for row in rows:

                if row.shape[1] < row_width:

                    pad = np.full(
                        (
                            row.shape[0],
                            row_width - row.shape[1],
                            3
                        ),
                        245,
                        dtype=np.uint8
                    )

                    row = np.hstack([
                        row,
                        pad
                    ])

                fixed_rows.append(
                    row
                )

            combined = np.vstack(
                fixed_rows
            )

        combined_path = os.path.join(
            OUTPUT_DIR,
            "TRINAY_XAI_ALL_LESIONS.png"
        )

        cv2.imwrite(
            combined_path,
            combined
        )

        print(
            f"✓ Combined XAI saved:"
        )

        print(
            f"  {combined_path}"
        )

    # ========================================================
    # FINAL CLEANUP
    # ========================================================

    del probabilities
    del input_tensor
    del model

    cleanup_memory()

    print(
        "\n"
        + "=" * 78
    )

    print(
        "                 TRINAY XAI COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        "\nXAI directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            'python src/xai.py '
            '"path/to/fundus_image.jpg"'
        )

        sys.exit(1)

    run_xai(
        sys.argv[1]
    )