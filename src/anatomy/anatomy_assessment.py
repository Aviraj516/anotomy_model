import cv2
import numpy as np


# ============================================================
# TRINAY - ANATOMY ASSESSMENT
# ============================================================


# ============================================================
# BLOOD VESSEL FEATURES
# ============================================================

def calculate_vessel_features(vessel_mask):

    vessel_mask = (vessel_mask > 0).astype(np.uint8)

    h, w = vessel_mask.shape

    total_pixels = h * w
    vessel_pixels = int(np.sum(vessel_mask))

    vessel_density = vessel_pixels / total_pixels

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        vessel_mask,
        connectivity=8
    )

    component_areas = []

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area > 5:
            component_areas.append(area)

    largest_component = (
        max(component_areas)
        if component_areas
        else 0
    )

    return {
        "vessel_pixels": vessel_pixels,
        "total_pixels": total_pixels,
        "vessel_density": float(vessel_density),
        "connected_components": len(component_areas),
        "largest_component_area": int(largest_component)
    }


# ============================================================
# OPTIC DISC FEATURES
# ============================================================

def calculate_optic_disc_features(optic_disc_mask):

    optic_disc_mask = (
        optic_disc_mask > 0
    ).astype(np.uint8)

    total_pixels = (
        optic_disc_mask.shape[0] *
        optic_disc_mask.shape[1]
    )

    disc_pixels = int(
        np.sum(optic_disc_mask)
    )

    disc_area_ratio = (
        disc_pixels / total_pixels
    )

    contours, _ = cv2.findContours(
        optic_disc_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:

        return {
            "detected": False,
            "area_pixels": 0,
            "area_ratio": 0.0,
            "centroid_x": None,
            "centroid_y": None,
            "bbox_x": None,
            "bbox_y": None,
            "bbox_width": None,
            "bbox_height": None,
            "circularity": 0.0
        }

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter > 0:

        circularity = (
            4 * np.pi * area
        ) / (
            perimeter ** 2
        )

    else:

        circularity = 0.0

    x, y, width, height = cv2.boundingRect(
        contour
    )

    moments = cv2.moments(contour)

    if moments["m00"] != 0:

        centroid_x = (
            moments["m10"] /
            moments["m00"]
        )

        centroid_y = (
            moments["m01"] /
            moments["m00"]
        )

    else:

        centroid_x = x + width / 2
        centroid_y = y + height / 2

    return {
        "detected": True,
        "area_pixels": int(area),
        "area_ratio": float(disc_area_ratio),
        "centroid_x": float(centroid_x),
        "centroid_y": float(centroid_y),
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_width": int(width),
        "bbox_height": int(height),
        "circularity": float(circularity)
    }


# ============================================================
# FOVEA FEATURES
# ============================================================

def calculate_fovea_features(
    fovea_x,
    fovea_y,
    image_width,
    image_height
):

    x_normalized = (
        fovea_x / image_width
    )

    y_normalized = (
        fovea_y / image_height
    )

    return {
        "x": float(fovea_x),
        "y": float(fovea_y),
        "x_normalized": float(x_normalized),
        "y_normalized": float(y_normalized)
    }


# ============================================================
# DISC-FOVEA RELATIONSHIP
# ============================================================

def calculate_disc_fovea_relationship(
    disc_x,
    disc_y,
    fovea_x,
    fovea_y
):

    distance = np.sqrt(
        (fovea_x - disc_x) ** 2 +
        (fovea_y - disc_y) ** 2
    )

    return {
        "disc_fovea_distance_pixels": float(
            distance
        )
    }


# ============================================================
# MACULA FEATURES
# ============================================================

def calculate_macula_features(
    image,
    fovea_x,
    fovea_y
):

    """
    Macula proxy assessment.

    Uses a region around the predicted fovea.
    This is NOT a macular disease detector.
    """

    h, w = image.shape[:2]

    radius = int(
        min(h, w) * 0.12
    )

    x = int(
        np.clip(
            fovea_x,
            0,
            w - 1
        )
    )

    y = int(
        np.clip(
            fovea_y,
            0,
            h - 1
        )
    )

    x1 = max(0, x - radius)
    x2 = min(w, x + radius)

    y1 = max(0, y - radius)
    y2 = min(h, y + radius)

    roi = image[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:

        return {
            "roi_mean_intensity": 0.0,
            "roi_std_intensity": 0.0
        }

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_RGB2GRAY
    )

    return {
        "roi_mean_intensity": float(
            np.mean(gray)
        ),
        "roi_std_intensity": float(
            np.std(gray)
        )
    }


# ============================================================
# RETINAL BACKGROUND FEATURES
# ============================================================

def calculate_background_features(
    image,
    vessel_mask,
    optic_disc_mask
):

    """
    Calculates basic retinal background statistics.

    This is a quality/appearance proxy.
    It is NOT a retinal disease classifier.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    valid_mask = np.ones(
        gray.shape,
        dtype=np.uint8
    )

    # Remove vessels
    valid_mask[
        vessel_mask > 0
    ] = 0

    # Remove optic disc
    valid_mask[
        optic_disc_mask > 0
    ] = 0

    pixels = gray[
        valid_mask > 0
    ]

    if len(pixels) == 0:

        return {
            "mean_intensity": 0.0,
            "std_intensity": 0.0
        }

    return {
        "mean_intensity": float(
            np.mean(pixels)
        ),
        "std_intensity": float(
            np.std(pixels)
        )
    }


# ============================================================
# FEATURE COLLECTION
# ============================================================

def calculate_anatomy_features(
    vessel_mask,
    optic_disc_mask,
    fovea_x,
    fovea_y,
    image_width,
    image_height,
    image=None
):

    vessel_features = calculate_vessel_features(
        vessel_mask
    )

    optic_disc_features = calculate_optic_disc_features(
        optic_disc_mask
    )

    fovea_features = calculate_fovea_features(
        fovea_x,
        fovea_y,
        image_width,
        image_height
    )

    if optic_disc_features["detected"]:

        relationship = (
            calculate_disc_fovea_relationship(
                optic_disc_features["centroid_x"],
                optic_disc_features["centroid_y"],
                fovea_x,
                fovea_y
            )
        )

    else:

        relationship = {
            "disc_fovea_distance_pixels": None
        }

    # --------------------------------------------------------
    # Macula
    # --------------------------------------------------------

    if image is not None:

        macula_features = (
            calculate_macula_features(
                image,
                fovea_x,
                fovea_y
            )
        )

        background_features = (
            calculate_background_features(
                image,
                vessel_mask,
                optic_disc_mask
            )
        )

    else:

        macula_features = {
            "roi_mean_intensity": None,
            "roi_std_intensity": None
        }

        background_features = {
            "mean_intensity": None,
            "std_intensity": None
        }

    return {

        "blood_vessels": vessel_features,

        "optic_disc": optic_disc_features,

        "fovea": fovea_features,

        "disc_fovea_relationship": relationship,

        "macula": macula_features,

        "retinal_background": background_features

    }


# ============================================================
# PROTOTYPE ASSESSMENT
# ============================================================

def assess_anatomy(results):

    assessments = {}

    # ========================================================
    # BLOOD VESSELS
    # ========================================================

    density = results[
        "blood_vessels"
    ]["vessel_density"]

    if density < 0.03 or density > 0.35:

        vessel_status = "Abnormal"

    elif 0.05 <= density <= 0.25:

        vessel_status = "Normal"

    else:

        vessel_status = "Uncertain"

    assessments["blood_vessels"] = vessel_status

    # ========================================================
    # OPTIC DISC
    # ========================================================

    disc = results[
        "optic_disc"
    ]

    if not disc["detected"]:

        disc_status = "Abnormal"

    elif (
        disc["area_ratio"] < 0.005
        or
        disc["area_ratio"] > 0.15
    ):

        disc_status = "Uncertain"

    else:

        disc_status = "Normal"

    assessments["optic_disc"] = disc_status

    # ========================================================
    # FOVEA
    # ========================================================

    fovea = results["fovea"]

    x = fovea["x_normalized"]
    y = fovea["y_normalized"]

    if (
        0.05 <= x <= 0.95
        and
        0.05 <= y <= 0.95
    ):

        fovea_status = "Normal"

    else:

        fovea_status = "Uncertain"

    assessments["fovea"] = fovea_status

    # ========================================================
    # MACULA
    # ========================================================

    macula = results["macula"]

    if (
        macula["roi_mean_intensity"] is None
    ):

        macula_status = "Uncertain"

    elif (
        30 <= macula["roi_mean_intensity"] <= 220
    ):

        macula_status = "Normal"

    else:

        macula_status = "Uncertain"

    assessments["macula"] = macula_status

    # ========================================================
    # RETINAL BACKGROUND
    # ========================================================

    background = results[
        "retinal_background"
    ]

    if (
        background["mean_intensity"] is None
    ):

        background_status = "Uncertain"

    elif (
        30 <= background["mean_intensity"] <= 220
    ):

        background_status = "Normal"

    else:

        background_status = "Uncertain"

    assessments[
        "retinal_background"
    ] = background_status

    return assessments


# ============================================================
# PRINT REPORT
# ============================================================

def print_anatomy_report(results):

    assessments = assess_anatomy(
        results
    )

    print()
    print("=" * 70)
    print("                 TRINAY ANATOMY ASSESSMENT")
    print("=" * 70)

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print("FINAL ANATOMY STATUS")
    print("-" * 70)

    print(
        f"Blood Vessels      : "
        f"{assessments['blood_vessels']}"
    )

    print(
        f"Optic Disc         : "
        f"{assessments['optic_disc']}"
    )

    print(
        f"Fovea              : "
        f"{assessments['fovea']}"
    )

    print(
        f"Macula             : "
        f"{assessments['macula']}"
    )

    print(
        f"Retinal Background : "
        f"{assessments['retinal_background']}"
    )

    # ========================================================
    # BLOOD VESSELS
    # ========================================================

    vessels = results[
        "blood_vessels"
    ]

    print()
    print("BLOOD VESSEL FEATURES")
    print("-" * 70)

    print(
        f"Vessel pixels        : "
        f"{vessels['vessel_pixels']}"
    )

    print(
        f"Vessel density       : "
        f"{vessels['vessel_density']:.4f}"
    )

    print(
        f"Connected components : "
        f"{vessels['connected_components']}"
    )

    print(
        f"Largest component    : "
        f"{vessels['largest_component_area']}"
    )

    # ========================================================
    # OPTIC DISC
    # ========================================================

    disc = results[
        "optic_disc"
    ]

    print()
    print("OPTIC DISC FEATURES")
    print("-" * 70)

    print(
        f"Detected     : "
        f"{disc['detected']}"
    )

    print(
        f"Area         : "
        f"{disc['area_pixels']} pixels"
    )

    print(
        f"Area ratio   : "
        f"{disc['area_ratio']:.6f}"
    )

    print(
        f"Circularity  : "
        f"{disc['circularity']:.4f}"
    )

    if disc["detected"]:

        print(
            f"Centroid     : "
            f"({disc['centroid_x']:.1f}, "
            f"{disc['centroid_y']:.1f})"
        )

    # ========================================================
    # FOVEA
    # ========================================================

    fovea = results[
        "fovea"
    ]

    print()
    print("FOVEA FEATURES")
    print("-" * 70)

    print(
        f"Location     : "
        f"({fovea['x']:.1f}, "
        f"{fovea['y']:.1f})"
    )

    print(
        f"Normalized   : "
        f"({fovea['x_normalized']:.4f}, "
        f"{fovea['y_normalized']:.4f})"
    )

    # ========================================================
    # DISC-FOVEA
    # ========================================================

    relationship = results[
        "disc_fovea_relationship"
    ]

    print()
    print("DISC - FOVEA RELATIONSHIP")
    print("-" * 70)

    if relationship[
        "disc_fovea_distance_pixels"
    ] is not None:

        print(
            f"Distance     : "
            f"{relationship['disc_fovea_distance_pixels']:.2f} pixels"
        )

    else:

        print(
            "Distance     : Not available"
        )

    # ========================================================
    # MACULA
    # ========================================================

    macula = results[
        "macula"
    ]

    print()
    print("MACULA FEATURES")
    print("-" * 70)

    print(
        f"Mean intensity : "
        f"{macula['roi_mean_intensity']}"
    )

    print(
        f"Intensity std  : "
        f"{macula['roi_std_intensity']}"
    )

    # ========================================================
    # BACKGROUND
    # ========================================================

    background = results[
        "retinal_background"
    ]

    print()
    print("RETINAL BACKGROUND")
    print("-" * 70)

    print(
        f"Mean intensity : "
        f"{background['mean_intensity']}"
    )

    print(
        f"Intensity std  : "
        f"{background['std_intensity']}"
    )

    print()
    print("=" * 70)