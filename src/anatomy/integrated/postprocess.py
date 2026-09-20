import cv2
import numpy as np


# ============================================================
# VISUAL POST-PROCESSING
# ============================================================
#
# IMPORTANT:
# These operations are ONLY for the displayed overlay.
# They do not modify model weights or benchmark metrics.
#
# Different lesions get different treatment because MA/HE
# are tiny while EX/SE are generally larger.
# ============================================================


# Conservative minimum areas.
# These are deliberately small so MA/HE are not destroyed.
MIN_AREA = {
    0: 2,      # MA
    1: 4,      # HE
    2: 12,     # EX
    3: 6       # SE
}


def remove_small_components(mask, min_area):
    """
    Remove connected components smaller than min_area.

    Input:
        mask: binary uint8 mask, values 0/1 or 0/255

    Output:
        cleaned binary mask, values 0/255
    """

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Normalize to 0/255
    if mask.max() <= 1:
        mask = mask * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:

            cleaned[
                labels == label
            ] = 255

    return cleaned


def fill_small_holes(mask, max_hole_area=30):
    """
    Fill small internal holes.

    Useful for EX/SE visualization.

    Large structures are left untouched.
    """

    mask = mask.astype(np.uint8)

    if mask.max() <= 1:
        mask = mask * 255

    # Invert mask
    inverted = cv2.bitwise_not(mask)

    # Flood fill background
    flood = inverted.copy()

    h, w = flood.shape

    flood_mask = np.zeros(
        (h + 2, w + 2),
        np.uint8
    )

    cv2.floodFill(
        flood,
        flood_mask,
        (0, 0),
        255
    )

    holes = cv2.bitwise_not(
        flood
    )

    # Connected components in holes
    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            holes,
            connectivity=8
        )
    )

    result = mask.copy()

    for label in range(1, num_labels):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area <= max_hole_area:

            result[
                labels == label
            ] = 255

    return result


def smooth_mask(mask):
    """
    Very mild boundary smoothing.
    """

    mask = mask.astype(np.uint8)

    if mask.max() <= 1:
        mask = mask * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    return mask


def process_single_mask(
    mask,
    class_idx
):
    """
    Class-specific post-processing.
    """

    min_area = MIN_AREA[
        class_idx
    ]

    cleaned = remove_small_components(
        mask,
        min_area
    )

    # --------------------------------------------------------
    # MA
    # --------------------------------------------------------
    #
    # Keep MA conservative.
    # Do NOT aggressively blur or close tiny lesions.
    #

    if class_idx == 0:

        return cleaned


    # --------------------------------------------------------
    # HE
    # --------------------------------------------------------

    if class_idx == 1:

        cleaned = smooth_mask(
            cleaned
        )

        return cleaned


    # --------------------------------------------------------
    # EX
    # --------------------------------------------------------

    if class_idx == 2:

        cleaned = smooth_mask(
            cleaned
        )

        cleaned = fill_small_holes(
            cleaned,
            max_hole_area=50
        )

        return cleaned


    # --------------------------------------------------------
    # SE
    # --------------------------------------------------------

    if class_idx == 3:

        cleaned = smooth_mask(
            cleaned
        )

        cleaned = fill_small_holes(
            cleaned,
            max_hole_area=35
        )

        return cleaned


    return cleaned


def postprocess_masks(
    masks,
    num_classes=4
):
    """
    Process all four lesion masks.

    Parameters
    ----------
    masks : dict
        {
            0: MA mask,
            1: HE mask,
            2: EX mask,
            3: SE mask
        }

    Returns
    -------
    processed_masks : dict
        Cleaned binary masks in 0/255 format.
    """

    processed_masks = {}

    for class_idx in range(num_classes):

        processed_masks[class_idx] = (
            process_single_mask(
                masks[class_idx],
                class_idx
            )
        )

    return processed_masks


def get_regions(mask):
    """
    Return connected lesion regions sorted by area.
    """

    if mask.max() <= 1:
        mask = mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area <= 0:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        regions.append({
            "contour": contour,
            "area": area,
            "x": x,
            "y": y,
            "w": w,
            "h": h
        })

    regions.sort(
        key=lambda r: r["area"],
        reverse=True
    )

    return regions


def summarize_masks(
    masks
):
    """
    Produce a simple summary for the nurse UI.
    """

    names = {
        0: "MA",
        1: "HE",
        2: "EX",
        3: "SE"
    }

    summary = {}

    for class_idx in range(4):

        regions = get_regions(
            masks[class_idx]
        )

        pixel_count = int(
            np.count_nonzero(
                masks[class_idx]
            )
        )

        summary[
            names[class_idx]
        ] = {
            "detected": (
                pixel_count > 0
            ),
            "regions": len(regions),
            "pixels": pixel_count
        }

    return summary