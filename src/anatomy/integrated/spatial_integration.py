import numpy as np
import cv2


# ============================================================
# TRINAY - SPATIAL INTEGRATION
# ============================================================
#
# Purpose:
# Combine anatomical predictions spatially:
#
#   1. Blood vessels
#   2. Optic disc
#   3. Fovea
#   4. Macular ROI
#   5. Lesion locations
#
# IMPORTANT:
# This module performs spatial/anatomical analysis.
# It does NOT independently diagnose disease.
# Clinical thresholds must be validated before being
# interpreted as medical normal/abnormal decisions.
# ============================================================


# ============================================================
# 1. CREATE MACULAR REGION AROUND FOVEA
# ============================================================

def create_fovea_region(
    fovea_x,
    fovea_y,
    image_shape,
    radius_ratio=0.12
):
    """
    Create a circular region centered on the predicted fovea.

    This region is used as a proxy for the macular region.

    Parameters
    ----------
    fovea_x : float
        Fovea X-coordinate in image pixels.

    fovea_y : float
        Fovea Y-coordinate in image pixels.

    image_shape : tuple
        Image shape, normally (H, W).

    radius_ratio : float
        Radius as a fraction of the smaller image dimension.

    Returns
    -------
    numpy.ndarray
        Binary macular ROI mask.
    """

    h, w = image_shape[:2]

    radius = max(
        1,
        int(min(h, w) * radius_ratio)
    )

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    # Keep the center inside image boundaries
    center_x = int(
        np.clip(fovea_x, 0, w - 1)
    )

    center_y = int(
        np.clip(fovea_y, 0, h - 1)
    )

    cv2.circle(
        mask,
        (center_x, center_y),
        radius,
        1,
        -1
    )

    return mask


# ============================================================
# 2. OPTIC DISC - FOVEA RELATIONSHIP
# ============================================================

def calculate_disc_fovea_distance(
    optic_disc_mask,
    fovea_x,
    fovea_y
):
    """
    Calculate the distance between the optic disc center
    and the predicted fovea center.

    Parameters
    ----------
    optic_disc_mask : numpy.ndarray
        Binary optic disc segmentation mask.

    fovea_x : float
        Fovea X-coordinate.

    fovea_y : float
        Fovea Y-coordinate.

    Returns
    -------
    dict or None
        Disc center, fovea center and Euclidean distance.
    """

    optic_disc_mask = (
        optic_disc_mask > 0
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        optic_disc_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Select largest detected optic disc region
    contour = max(
        contours,
        key=cv2.contourArea
    )

    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None

    disc_x = (
        moments["m10"] /
        moments["m00"]
    )

    disc_y = (
        moments["m01"] /
        moments["m00"]
    )

    distance = np.sqrt(
        (fovea_x - disc_x) ** 2 +
        (fovea_y - disc_y) ** 2
    )

    return {
        "optic_disc_center": (
            float(disc_x),
            float(disc_y)
        ),

        "fovea_center": (
            float(fovea_x),
            float(fovea_y)
        ),

        "distance_pixels": float(
            distance
        )
    }


# ============================================================
# 3. BLOOD VESSEL ANALYSIS
# ============================================================

def analyze_vessels(
    vessel_mask,
    macula_mask=None
):
    """
    Analyze the predicted blood vessel segmentation.

    Calculates:

        - total pixels
        - vessel pixels
        - global vessel density
        - connected components
        - largest vessel component
        - macular vessel density

    Parameters
    ----------
    vessel_mask : numpy.ndarray
        Binary vessel mask.

    macula_mask : numpy.ndarray, optional
        Binary macular ROI mask.

    Returns
    -------
    dict
        Vessel spatial features.
    """

    vessel_mask = (
        vessel_mask > 0
    ).astype(np.uint8)

    total_pixels = vessel_mask.size

    vessel_pixels = int(
        np.sum(vessel_mask)
    )

    # --------------------------------------------------------
    # Global vessel density
    # --------------------------------------------------------

    if total_pixels > 0:

        density = (
            vessel_pixels /
            total_pixels
        )

    else:

        density = 0.0

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            vessel_mask,
            connectivity=8
        )
    )

    component_sizes = []

    for i in range(1, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        # Ignore tiny noise components
        if area > 5:

            component_sizes.append(
                int(area)
            )

    connected_components = len(
        component_sizes
    )

    largest_component = (
        max(component_sizes)
        if component_sizes
        else 0
    )

    # --------------------------------------------------------
    # Macular vessel density
    # --------------------------------------------------------

    macula_density = None

    if macula_mask is not None:

        macula_mask = (
            macula_mask > 0
        ).astype(np.uint8)

        # Check shape compatibility
        if macula_mask.shape == vessel_mask.shape:

            macula_pixels = (
                macula_mask > 0
            )

            macula_area = int(
                np.sum(macula_pixels)
            )

            if macula_area > 0:

                macula_vessels = int(
                    np.sum(
                        vessel_mask[
                            macula_pixels
                        ]
                    )
                )

                macula_density = (
                    macula_vessels /
                    macula_area
                )

    return {
        "total_pixels": int(
            total_pixels
        ),

        "vessel_pixels": int(
            vessel_pixels
        ),

        "vessel_density": float(
            density
        ),

        "connected_components": int(
            connected_components
        ),

        "largest_component": int(
            largest_component
        ),

        "macula_vessel_density": (
            float(macula_density)
            if macula_density is not None
            else None
        )
    }


# ============================================================
# 4. LESION LOCATION ANALYSIS
# ============================================================

def analyze_lesion_locations(
    lesion_mask,
    macula_mask
):
    """
    Analyze the spatial relationship between lesions
    and the macular region.

    Parameters
    ----------
    lesion_mask : numpy.ndarray
        Binary lesion mask.

    macula_mask : numpy.ndarray
        Binary macular ROI mask.

    Returns
    -------
    dict
        Lesion spatial features.

    Notes
    -----
    This function does not classify the lesion type.
    It only determines how much of the lesion mask lies
    inside the macular ROI.
    """

    lesion_mask = (
        lesion_mask > 0
    ).astype(np.uint8)

    macula_mask = (
        macula_mask > 0
    ).astype(np.uint8)

    # Make sure both masks have the same shape
    if lesion_mask.shape != macula_mask.shape:

        raise ValueError(
            "lesion_mask and macula_mask "
            "must have the same shape."
        )

    lesion_pixels = int(
        np.sum(lesion_mask)
    )

    # --------------------------------------------------------
    # No lesion detected
    # --------------------------------------------------------

    if lesion_pixels == 0:

        return {
            "total_lesion_pixels": 0,

            "macula_lesion_pixels": 0,

            "macula_overlap_ratio": 0.0
        }

    # --------------------------------------------------------
    # Lesion pixels inside macular ROI
    # --------------------------------------------------------

    macula_lesion_pixels = int(
        np.sum(
            lesion_mask[
                macula_mask > 0
            ]
        )
    )

    # --------------------------------------------------------
    # Percentage of lesion pixels located
    # inside the macular region
    # --------------------------------------------------------

    overlap_ratio = (
        macula_lesion_pixels /
        lesion_pixels
    )

    return {
        "total_lesion_pixels": int(
            lesion_pixels
        ),

        "macula_lesion_pixels": int(
            macula_lesion_pixels
        ),

        "macula_overlap_ratio": float(
            overlap_ratio
        )
    }


# ============================================================
# 5. COMPLETE SPATIAL ANALYSIS
# ============================================================

def build_spatial_analysis(
    vessel_mask,
    optic_disc_mask,
    fovea_x,
    fovea_y
):
    """
    Build the complete anatomical spatial analysis.

    Inputs
    ------

    vessel_mask:
        Predicted blood vessel mask.

    optic_disc_mask:
        Predicted optic disc mask.

    fovea_x:
        Predicted fovea X-coordinate.

    fovea_y:
        Predicted fovea Y-coordinate.

    Returns
    -------

    dict containing:

        macula_mask
        vessel_analysis
        disc_fovea
    """

    # --------------------------------------------------------
    # Validate vessel mask
    # --------------------------------------------------------

    vessel_mask = (
        vessel_mask > 0
    ).astype(np.uint8)

    if vessel_mask.ndim != 2:

        raise ValueError(
            "vessel_mask must be a 2D binary mask."
        )

    h, w = vessel_mask.shape

    # --------------------------------------------------------
    # Validate optic disc mask
    # --------------------------------------------------------

    optic_disc_mask = (
        optic_disc_mask > 0
    ).astype(np.uint8)

    if optic_disc_mask.shape != (
        h,
        w
    ):

        raise ValueError(
            "optic_disc_mask and vessel_mask "
            "must have the same shape."
        )

    # --------------------------------------------------------
    # 1. Create macular ROI
    # --------------------------------------------------------

    macula_mask = create_fovea_region(
        fovea_x=fovea_x,
        fovea_y=fovea_y,
        image_shape=(h, w),
        radius_ratio=0.12
    )

    # --------------------------------------------------------
    # 2. Analyze blood vessels
    # --------------------------------------------------------

    vessel_features = analyze_vessels(
        vessel_mask=vessel_mask,
        macula_mask=macula_mask
    )

    # --------------------------------------------------------
    # 3. Calculate optic disc-fovea relationship
    # --------------------------------------------------------

    disc_fovea = calculate_disc_fovea_distance(
        optic_disc_mask=optic_disc_mask,
        fovea_x=fovea_x,
        fovea_y=fovea_y
    )

    # --------------------------------------------------------
    # 4. Return complete spatial analysis
    # --------------------------------------------------------

    return {

        "image_shape": (
            int(h),
            int(w)
        ),

        "fovea_center": (
            float(fovea_x),
            float(fovea_y)
        ),

        "macula_mask": macula_mask,

        "vessel_analysis": vessel_features,

        "disc_fovea": disc_fovea
    }


# ============================================================
# 6. OPTIONAL HELPER FOR LESION INTEGRATION
# ============================================================

def add_lesion_spatial_analysis(
    spatial_results,
    lesion_mask
):
    """
    Add lesion-to-macula spatial analysis to an existing
    spatial analysis result.

    This function is intended for the next stage of TRINAY,
    when the existing lesion model is connected.

    Parameters
    ----------
    spatial_results : dict
        Output from build_spatial_analysis().

    lesion_mask : numpy.ndarray
        Binary lesion mask from the existing lesion model.

    Returns
    -------
    dict
        Updated spatial analysis.
    """

    if "macula_mask" not in spatial_results:

        raise ValueError(
            "spatial_results does not contain "
            "'macula_mask'."
        )

    macula_mask = spatial_results[
        "macula_mask"
    ]

    lesion_features = analyze_lesion_locations(
        lesion_mask=lesion_mask,
        macula_mask=macula_mask
    )

    spatial_results[
        "lesion_analysis"
    ] = lesion_features

    return spatial_results