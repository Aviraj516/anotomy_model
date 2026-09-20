import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


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


# ============================================================
# MODEL IMPORTS
# ============================================================

from src.anatomy.models.unet_efficientnet import VesselUNet
from src.anatomy.models.optic_disc_unet import OpticDiscUNet
from src.anatomy.models.fovea_net import FoveaNet


# ============================================================
# ANATOMY ASSESSMENT
# ============================================================

from src.anatomy.anatomy_assessment import (
    calculate_anatomy_features,
    print_anatomy_report
)


# ============================================================
# SPATIAL INTEGRATION
# ============================================================

from src.anatomy.integrated.spatial_integration import (
    build_spatial_analysis
)


# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = 512


VESSEL_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "anatomy",
    "vessels",
    "best_vessel_unet.pth"
)


OPTIC_MODEL_PATH = os.path.join(
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


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "anatomy",
    "combined"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


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
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not read image:\n{image_path}"
        )

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original_h, original_w = image_rgb.shape[:2]

    resized = cv2.resize(
        image_rgb,
        (IMG_SIZE, IMG_SIZE)
    )

    normalized = (
        resized.astype(
            np.float32
        ) / 255.0
    )

    tensor = torch.tensor(
        normalized,
        dtype=torch.float32
    ).permute(
        2,
        0,
        1
    ).unsqueeze(0)

    return (
        image_rgb,
        tensor,
        original_w,
        original_h
    )


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    print("\nLoading anatomy models...")

    # --------------------------------------------------------
    # Vessel
    # --------------------------------------------------------

    vessel_model = VesselUNet().to(device)

    vessel_checkpoint = torch.load(
        VESSEL_MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" in vessel_checkpoint:

        vessel_model.load_state_dict(
            vessel_checkpoint["model_state_dict"]
        )

    else:

        vessel_model.load_state_dict(
            vessel_checkpoint
        )

    vessel_model.eval()

    print("✓ Vessel model loaded")

    # --------------------------------------------------------
    # Optic Disc
    # --------------------------------------------------------

    optic_model = OpticDiscUNet().to(device)

    optic_checkpoint = torch.load(
        OPTIC_MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" in optic_checkpoint:

        optic_model.load_state_dict(
            optic_checkpoint["model_state_dict"]
        )

    else:

        optic_model.load_state_dict(
            optic_checkpoint
        )

    optic_model.eval()

    print("✓ Optic disc model loaded")

    # --------------------------------------------------------
    # Fovea
    # --------------------------------------------------------

    fovea_model = FoveaNet().to(device)

    fovea_checkpoint = torch.load(
        FOVEA_MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" in fovea_checkpoint:

        fovea_model.load_state_dict(
            fovea_checkpoint["model_state_dict"]
        )

    else:

        fovea_model.load_state_dict(
            fovea_checkpoint
        )

    fovea_model.eval()

    print("✓ Fovea model loaded")

    return (
        vessel_model,
        optic_model,
        fovea_model
    )


# ============================================================
# VESSEL PREDICTION
# ============================================================

def predict_vessels(
    model,
    image_tensor
):

    with torch.no_grad():

        output = model(
            image_tensor.to(device)
        )

        probability = torch.sigmoid(
            output
        )

    probability = (
        probability[0, 0]
        .cpu()
        .numpy()
    )

    mask = (
        probability > 0.5
    ).astype(
        np.uint8
    )

    return (
        probability,
        mask
    )


# ============================================================
# OPTIC DISC PREDICTION
# ============================================================

def predict_optic_disc(
    model,
    image_tensor
):

    with torch.no_grad():

        output = model(
            image_tensor.to(device)
        )

        probability = torch.sigmoid(
            output
        )

    probability = (
        probability[0, 0]
        .cpu()
        .numpy()
    )

    mask = (
        probability > 0.5
    ).astype(
        np.uint8
    )

    return (
        probability,
        mask
    )


# ============================================================
# FOVEA PREDICTION
# ============================================================

def predict_fovea(
    model,
    image_tensor,
    original_w,
    original_h
):

    with torch.no_grad():

        prediction = model(
            image_tensor.to(device)
        )

    prediction = (
        prediction[0]
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Normalized → original image coordinates
    # --------------------------------------------------------

    x = prediction[0] * original_w
    y = prediction[1] * original_h

    # --------------------------------------------------------
    # Normalized → 512x512 coordinates
    # --------------------------------------------------------

    x_512 = prediction[0] * IMG_SIZE
    y_512 = prediction[1] * IMG_SIZE

    return (
        x,
        y,
        x_512,
        y_512
    )


# ============================================================
# DRAW ANATOMY OVERLAY
# ============================================================

def create_overlay(
    image,
    vessel_mask,
    optic_mask,
    fovea_x,
    fovea_y
):

    overlay = image.copy()

    # --------------------------------------------------------
    # Resize masks to original image
    # --------------------------------------------------------

    h, w = image.shape[:2]

    vessel_mask = cv2.resize(
        vessel_mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    optic_mask = cv2.resize(
        optic_mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # Vessel overlay
    # --------------------------------------------------------

    vessel_pixels = (
        vessel_mask > 0
    )

    overlay[
        vessel_pixels
    ] = (
        255,
        80,
        80
    )

    # --------------------------------------------------------
    # Optic disc contour
    # --------------------------------------------------------

    optic_uint8 = (
        optic_mask * 255
    ).astype(
        np.uint8
    )

    contours, _ = cv2.findContours(
        optic_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (80, 255, 80),
        5
    )

    # --------------------------------------------------------
    # Fovea point
    # --------------------------------------------------------

    cv2.circle(
        overlay,
        (
            int(fovea_x),
            int(fovea_y)
        ),
        15,
        (80, 120, 255),
        -1
    )

    cv2.circle(
        overlay,
        (
            int(fovea_x),
            int(fovea_y)
        ),
        25,
        (255, 255, 255),
        3
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    cv2.putText(
        overlay,
        "Optic Disc",
        (
            30,
            45
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (80, 255, 80),
        2
    )

    cv2.putText(
        overlay,
        "Fovea",
        (
            int(fovea_x) + 20,
            int(fovea_y)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 120, 255),
        2
    )

    cv2.putText(
        overlay,
        "Blood Vessels",
        (
            30,
            85
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 80, 80),
        2
    )

    return overlay


# ============================================================
# CREATE MACULA OVERLAY
# ============================================================

def create_spatial_overlay(
    image,
    vessel_mask,
    optic_mask,
    macula_mask,
    fovea_x,
    fovea_y
):

    overlay = image.copy()

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # Resize masks
    # --------------------------------------------------------

    vessel_mask = cv2.resize(
        vessel_mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    optic_mask = cv2.resize(
        optic_mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    macula_mask = cv2.resize(
        macula_mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # Macula ROI boundary
    # --------------------------------------------------------

    macula_uint8 = (
        macula_mask * 255
    ).astype(
        np.uint8
    )

    macula_contours, _ = cv2.findContours(
        macula_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        macula_contours,
        -1,
        (255, 255, 0),
        4
    )

    # --------------------------------------------------------
    # Blood vessels
    # --------------------------------------------------------

    vessel_pixels = (
        vessel_mask > 0
    )

    overlay[
        vessel_pixels
    ] = (
        255,
        80,
        80
    )

    # --------------------------------------------------------
    # Optic disc
    # --------------------------------------------------------

    optic_uint8 = (
        optic_mask * 255
    ).astype(
        np.uint8
    )

    optic_contours, _ = cv2.findContours(
        optic_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        optic_contours,
        -1,
        (80, 255, 80),
        5
    )

    # --------------------------------------------------------
    # Fovea
    # --------------------------------------------------------

    cv2.circle(
        overlay,
        (
            int(fovea_x),
            int(fovea_y)
        ),
        15,
        (80, 120, 255),
        -1
    )

    cv2.circle(
        overlay,
        (
            int(fovea_x),
            int(fovea_y)
        ),
        25,
        (255, 255, 255),
        3
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    cv2.putText(
        overlay,
        "Macular ROI",
        (
            30,
            125
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 0),
        2
    )

    return overlay


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    original,
    vessel_probability,
    vessel_mask,
    optic_probability,
    optic_mask,
    overlay,
    spatial_overlay,
    spatial_results,
    output_path
):

    # --------------------------------------------------------
    # Resize probability maps
    # --------------------------------------------------------

    h, w = original.shape[:2]

    vessel_probability = cv2.resize(
        vessel_probability,
        (w, h)
    )

    optic_probability = cv2.resize(
        optic_probability,
        (w, h)
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    plt.figure(
        figsize=(18, 12)
    )

    # --------------------------------------------------------
    # 1. Original
    # --------------------------------------------------------

    plt.subplot(
        2,
        4,
        1
    )

    plt.imshow(
        original
    )

    plt.title(
        "Original Fundus"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # 2. Vessel probability
    # --------------------------------------------------------

    plt.subplot(
        2,
        4,
        2
    )

    plt.imshow(
        vessel_probability,
        cmap="gray"
    )

    plt.title(
        "Blood Vessel Probability"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # 3. Vessel mask
    # --------------------------------------------------------

    plt.subplot(
        2,
        4,
        3
    )

    plt.imshow(
        vessel_mask,
        cmap="gray"
    )

    plt.title(
        "Blood Vessel Mask"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # 4. Optic disc probability
    # --------------------------------------------------------

    plt.subplot(
        2,
        4,
        4
    )

    plt.imshow(
        optic_probability,
        cmap="gray"
    )

    plt.title(
        "Optic Disc Probability"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # 5. Optic disc mask
    # --------------------------------------------------------

    plt.subplot(
        2,
        4,
        5
    )

    plt.imshow(
        optic_mask,
        cmap="gray"
    )

    plt.title(
        "Optic Disc Mask"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # 6. Anatomy overlay
    # --------------------------------------------------------

    plt.subplot(
        2,
        4,
        6
    )

    plt.imshow(
        overlay
    )

    plt.title(
        "Anatomy Overlay"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # 7. Spatial overlay
    # --------------------------------------------------------

    plt.subplot(
        2,
        4,
        7
    )

    plt.imshow(
        spatial_overlay
    )

    plt.title(
        "Spatial Anatomy"
    )

    plt.axis("off")

    # --------------------------------------------------------
    # 8. Spatial information
    # --------------------------------------------------------

    plt.subplot(
        2,
        4,
        8
    )

    plt.axis("off")

    vessel_features = spatial_results[
        "vessel_analysis"
    ]

    disc_fovea = spatial_results[
        "disc_fovea"
    ]

    text_lines = [
        "SPATIAL ANALYSIS",
        "",
        f"Vessel density:",
        f"{vessel_features['vessel_density']:.4f}",
        "",
        f"Macula vessel density:",
        f"{vessel_features['macula_vessel_density']:.4f}"
        if vessel_features[
            "macula_vessel_density"
        ] is not None
        else "N/A",
        "",
        f"Vessel components:",
        f"{vessel_features['connected_components']}",
    ]

    if disc_fovea is not None:

        text_lines.extend([
            "",
            "Disc-Fovea distance:",
            f"{disc_fovea['distance_pixels']:.2f} px"
        ])

    plt.text(
        0.05,
        0.95,
        "\n".join(text_lines),
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=11
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            'python src/anatomy/inference/'
            'predict_anatomy.py "image.jpg"'
        )

        sys.exit(1)

    image_path = sys.argv[1]

    print("\n" + "=" * 70)

    print(
        "                 TRINAY ANATOMY MODEL"
    )

    print("=" * 70)

    print(
        f"\nInput image:\n{image_path}"
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    (
        original,
        image_tensor,
        original_w,
        original_h
    ) = load_image(
        image_path
    )

    print(
        f"Original size: "
        f"{original_w} x {original_h}"
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    (
        vessel_model,
        optic_model,
        fovea_model
    ) = load_models()

    # --------------------------------------------------------
    # Vessel
    # --------------------------------------------------------

    print(
        "\nPredicting blood vessels..."
    )

    (
        vessel_probability,
        vessel_mask
    ) = predict_vessels(
        vessel_model,
        image_tensor
    )

    print(
        "✓ Blood vessels predicted"
    )

    # --------------------------------------------------------
    # Optic disc
    # --------------------------------------------------------

    print(
        "Predicting optic disc..."
    )

    (
        optic_probability,
        optic_mask
    ) = predict_optic_disc(
        optic_model,
        image_tensor
    )

    print(
        "✓ Optic disc predicted"
    )

    # --------------------------------------------------------
    # Fovea
    # --------------------------------------------------------

    print(
        "Predicting fovea..."
    )

    (
        fovea_x,
        fovea_y,
        fovea_x_512,
        fovea_y_512
    ) = predict_fovea(
        fovea_model,
        image_tensor,
        original_w,
        original_h
    )

    print(
        "✓ Fovea predicted"
    )

    # --------------------------------------------------------
    # Print predictions
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        "ANATOMY PREDICTIONS"
    )

    print("-" * 70)

    print(
        f"Fovea X : {fovea_x:.1f} px"
    )

    print(
        f"Fovea Y : {fovea_y:.1f} px"
    )

    print(
        f"Fovea (512x512): "
        f"({fovea_x_512:.1f}, "
        f"{fovea_y_512:.1f})"
    )

    print(
        f"Vessel pixels: "
        f"{np.sum(vessel_mask)}"
    )

    print(
        f"Optic disc pixels: "
        f"{np.sum(optic_mask)}"
    )

    # ========================================================
    # ANATOMY ASSESSMENT
    # ========================================================

    print(
        "\nCalculating anatomical features..."
    )

    # Vessel and optic-disc masks are 512x512.
    # Therefore use the 512x512 fovea coordinates.

    assessment_h, assessment_w = (
        vessel_mask.shape
    )

    assessment_image = cv2.resize(
        original,
        (
            assessment_w,
            assessment_h
        )
    )

    anatomy_results = calculate_anatomy_features(
        vessel_mask=vessel_mask,
        optic_disc_mask=optic_mask,
        fovea_x=fovea_x_512,
        fovea_y=fovea_y_512,
        image_width=assessment_w,
        image_height=assessment_h,
        image=assessment_image
    )

    print_anatomy_report(
        anatomy_results
    )

    # ========================================================
    # SPATIAL INTEGRATION
    # ========================================================

    print(
        "\nCalculating spatial anatomy..."
    )

    spatial_results = build_spatial_analysis(
        vessel_mask=vessel_mask,
        optic_disc_mask=optic_mask,
        fovea_x=fovea_x_512,
        fovea_y=fovea_y_512
    )

    print(
        "✓ Spatial integration complete"
    )

    # --------------------------------------------------------
    # Extract spatial features
    # --------------------------------------------------------

    vessel_features = spatial_results[
        "vessel_analysis"
    ]

    disc_fovea = spatial_results[
        "disc_fovea"
    ]

    print("\n" + "-" * 70)

    print(
        "SPATIAL ANALYSIS"
    )

    print("-" * 70)

    print(
        f"Global vessel density : "
        f"{vessel_features['vessel_density']:.4f}"
    )

    if vessel_features[
        "macula_vessel_density"
    ] is not None:

        print(
            f"Macular vessel density: "
            f"{vessel_features['macula_vessel_density']:.4f}"
        )

    else:

        print(
            "Macular vessel density: N/A"
        )

    print(
        f"Connected components  : "
        f"{vessel_features['connected_components']}"
    )

    print(
        f"Largest component     : "
        f"{vessel_features['largest_component']}"
    )

    if disc_fovea is not None:

        print(
            f"Disc-Fovea distance   : "
            f"{disc_fovea['distance_pixels']:.2f} px"
        )

    else:

        print(
            "Disc-Fovea distance   : N/A"
        )

    # ========================================================
    # CREATE ANATOMY OVERLAY
    # ========================================================

    overlay = create_overlay(
        original,
        vessel_mask,
        optic_mask,
        fovea_x,
        fovea_y
    )

    # ========================================================
    # CREATE SPATIAL OVERLAY
    # ========================================================

    spatial_overlay = create_spatial_overlay(
        original,
        vessel_mask,
        optic_mask,
        spatial_results["macula_mask"],
        fovea_x,
        fovea_y
    )

    # ========================================================
    # SAVE
    # ========================================================

    filename = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{filename}_anatomy.png"
    )

    save_result(
        original=original,
        vessel_probability=vessel_probability,
        vessel_mask=vessel_mask,
        optic_probability=optic_probability,
        optic_mask=optic_mask,
        overlay=overlay,
        spatial_overlay=spatial_overlay,
        spatial_results=spatial_results,
        output_path=output_path
    )

    print("\n" + "=" * 70)

    print(
        "ANATOMY + SPATIAL ANALYSIS COMPLETE"
    )

    print("=" * 70)

    print(
        f"\nOutput saved:\n{output_path}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()