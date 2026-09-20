import cv2
import numpy as np


# ============================================================
# TRINAY OVERLAY
# ============================================================
#
# This file controls VISUALIZATION ONLY.
#
# It does NOT change:
#   - model weights
#   - thresholds used for evaluation
#   - benchmark scores
#   - test results
#
# It only makes the predicted lesion map easier to understand.
# ============================================================


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


CLASS_DESCRIPTIONS = {
    0: "Small red retinal spots",
    1: "Bleeding / haemorrhage area",
    2: "Bright exudate deposits",
    3: "Soft exudate / cotton-wool area"
}


# ============================================================
# COLORS
# OpenCV uses BGR.
# ============================================================

COLORS = {

    # Red
    0: (45, 70, 235),

    # Violet / purple
    1: (185, 70, 190),

    # Yellow / gold
    2: (30, 205, 235),

    # Cyan / turquoise
    3: (220, 190, 35)
}


# ============================================================
# DISPLAY SETTINGS
# ============================================================

ALPHA = {
    0: 0.80,
    1: 0.36,
    2: 0.38,
    3: 0.36
}


# More aggressive DISPLAY cleanup for larger lesions.
# This is only visualization.
DISPLAY_MIN_AREA = {
    0: 2,
    1: 5,
    2: 18,
    3: 8
}


# ============================================================
# RETINA ENHANCEMENT
# ============================================================

def enhance_retina(image):
    """
    Mild enhancement for visualization.
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=1.25,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge(
        [l, a, b]
    )

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )

    # Very mild sharpening
    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1.0
    )

    enhanced = cv2.addWeighted(
        enhanced,
        1.05,
        blurred,
        -0.05,
        0
    )

    return enhanced


# ============================================================
# DISPLAY MASK CLEANUP
# ============================================================

def prepare_display_mask(
    mask,
    class_idx
):
    """
    Clean and slightly consolidate the prediction
    for visualization.

    IMPORTANT:
    This does NOT modify benchmark masks.
    """

    mask = mask.astype(
        np.uint8
    )

    if mask.max() <= 1:
        mask = mask * 255

    min_area = DISPLAY_MIN_AREA[
        class_idx
    ]

    # --------------------------------------------------------
    # Remove tiny components
    # --------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(
        mask
    )

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:

            cleaned[
                labels == label
            ] = 255

    # --------------------------------------------------------
    # Class-specific smoothing
    # --------------------------------------------------------

    if class_idx == 0:
        # MA is tiny: keep it conservative.
        return cleaned

    if class_idx == 1:
        # HE
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        )

        return cleaned

    if class_idx == 2:
        # EX
        #
        # Slightly stronger closing to join
        # nearby fragmented predictions.

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        )

        return cleaned

    if class_idx == 3:
        # SE

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        )

        return cleaned

    return cleaned


# ============================================================
# CONTOURS
# ============================================================

def get_contours(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )


# ============================================================
# REGION COUNT
# ============================================================

def count_regions(mask):

    num_labels, _, _, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    return max(
        0,
        num_labels - 1
    )


# ============================================================
# APPLY SOFT COLOR FILL
# ============================================================

def apply_colored_fill(
    image,
    mask,
    color,
    alpha
):
    """
    Semi-transparent lesion fill.
    """

    mask_bool = mask > 0

    if not np.any(mask_bool):
        return image

    color_layer = np.zeros_like(
        image
    )

    color_layer[:, :] = color

    result = image.copy()

    blended = cv2.addWeighted(
        image,
        1.0 - alpha,
        color_layer,
        alpha,
        0
    )

    result[
        mask_bool
    ] = blended[
        mask_bool
    ]

    return result


# ============================================================
# DRAW BOUNDARY
# ============================================================

def draw_boundary(
    image,
    mask,
    color,
    thickness=2
):

    contours = get_contours(
        mask
    )

    for contour in contours:

        if cv2.contourArea(
            contour
        ) < 2:

            continue

        # Slightly smooth contour
        perimeter = cv2.arcLength(
            contour,
            True
        )

        epsilon = (
            0.004 *
            perimeter
        )

        smooth = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        cv2.drawContours(
            image,
            [smooth],
            -1,
            color,
            thickness,
            cv2.LINE_AA
        )

    return image


# ============================================================
# MA MARKERS
# ============================================================

def draw_ma_markers(
    image,
    mask
):

    contours = get_contours(
        mask
    )

    color = COLORS[0]

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < 2:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        cx = x + w // 2
        cy = y + h // 2

        radius = max(
            3,
            min(
                5,
                int(
                    max(w, h) / 2
                ) + 1
            )
        )

        # Filled red dot
        cv2.circle(
            image,
            (cx, cy),
            radius,
            color,
            -1,
            cv2.LINE_AA
        )

        # Small dark center to separate nearby dots
        cv2.circle(
            image,
            (cx, cy),
            1,
            (40, 40, 40),
            -1,
            cv2.LINE_AA
        )

    return image


# ============================================================
# CREATE LESION OVERLAY
# ============================================================

def create_retina_overlay(
    image,
    masks
):
    """
    Build an organized lesion overlay.

    Drawing order:
        EX
        SE
        HE
        MA

    Larger regions are placed first,
    smaller lesions remain visible on top.
    """

    result = enhance_retina(
        image
    )

    # --------------------------------------------------------
    # Prepare masks
    # --------------------------------------------------------

    display_masks = {}

    for class_idx in range(4):

        display_masks[class_idx] = (
            prepare_display_mask(
                masks[class_idx],
                class_idx
            )
        )

    # --------------------------------------------------------
    # EX first
    # --------------------------------------------------------

    result = apply_colored_fill(
        result,
        display_masks[2],
        COLORS[2],
        ALPHA[2]
    )

    result = draw_boundary(
        result,
        display_masks[2],
        COLORS[2],
        thickness=2
    )

    # --------------------------------------------------------
    # SE
    # --------------------------------------------------------

    result = apply_colored_fill(
        result,
        display_masks[3],
        COLORS[3],
        ALPHA[3]
    )

    result = draw_boundary(
        result,
        display_masks[3],
        COLORS[3],
        thickness=2
    )

    # --------------------------------------------------------
    # HE
    # --------------------------------------------------------

    result = apply_colored_fill(
        result,
        display_masks[1],
        COLORS[1],
        ALPHA[1]
    )

    result = draw_boundary(
        result,
        display_masks[1],
        COLORS[1],
        thickness=2
    )

    # --------------------------------------------------------
    # MA LAST
    # --------------------------------------------------------

    result = draw_ma_markers(
        result,
        display_masks[0]
    )

    return result, display_masks


# ============================================================
# TITLE
# ============================================================

def add_title(
    image
):

    h, w = image.shape[:2]

    title_height = 54

    canvas = np.zeros(
        (
            h + title_height,
            w,
            3
        ),
        dtype=np.uint8
    )

    canvas[
        title_height:
    ] = image

    canvas[
        :title_height
    ] = (248, 248, 248)

    cv2.putText(
        canvas,
        "TRINAY",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (25, 25, 25),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        "AI-DETECTED RETINAL LESIONS",
        (125, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (80, 80, 80),
        1,
        cv2.LINE_AA
    )

    return canvas


# ============================================================
# DRAW COLOR CHIP
# ============================================================

def draw_color_chip(
    image,
    position,
    color
):

    x, y = position

    cv2.circle(
        image,
        (x, y),
        8,
        color,
        -1,
        cv2.LINE_AA
    )


# ============================================================
# SIDE INFORMATION PANEL
# ============================================================

def create_information_panel(
    summary,
    panel_width=360,
    panel_height=566
):

    panel = np.full(
        (
            panel_height,
            panel_width,
            3
        ),
        248,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    cv2.putText(
        panel,
        "LESION ANALYSIS",
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (30, 30, 30),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        panel,
        "AI-detected areas on the retina",
        (24, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (100, 100, 100),
        1,
        cv2.LINE_AA
    )

    # Divider
    cv2.line(
        panel,
        (20, 82),
        (panel_width - 20, 82),
        (215, 215, 215),
        1
    )

    # --------------------------------------------------------
    # Lesion cards
    # --------------------------------------------------------

    y = 110

    card_height = 82

    for class_idx in range(4):

        name = CLASS_NAMES[
            class_idx
        ]

        full_name = CLASS_FULL_NAMES[
            class_idx
        ]

        info = summary[
            name
        ]

        detected = info[
            "detected"
        ]

        regions = info[
            "regions"
        ]

        # card
        cv2.rectangle(
            panel,
            (18, y),
            (
                panel_width - 18,
                y + card_height
            ),
            (238, 238, 238),
            -1
        )

        # colored chip
        draw_color_chip(
            panel,
            (42, y + 28),
            COLORS[class_idx]
        )

        # abbreviation
        cv2.putText(
            panel,
            name,
            (62, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (30, 30, 30),
            2,
            cv2.LINE_AA
        )

        # full name
        cv2.putText(
            panel,
            full_name,
            (102, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (75, 75, 75),
            1,
            cv2.LINE_AA
        )

        # status
        status = (
            "Detected"
            if detected
            else "Not detected"
        )

        status_color = (
            (40, 150, 70)
            if detected
            else (110, 110, 110)
        )

        cv2.putText(
            panel,
            status,
            (62, y + 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            status_color,
            2,
            cv2.LINE_AA
        )

        # region count
        cv2.putText(
            panel,
            f"{regions} region"
            + ("" if regions == 1 else "s"),
            (220, y + 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (90, 90, 90),
            1,
            cv2.LINE_AA
        )

        y += 92

    # --------------------------------------------------------
    # Explanation box
    # --------------------------------------------------------

    box_y = y + 4

    cv2.rectangle(
        panel,
        (18, box_y),
        (
            panel_width - 18,
            panel_height - 24
        ),
        (232, 240, 248),
        -1
    )

    cv2.putText(
        panel,
        "WHAT THE MARKINGS MEAN",
        (32, box_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (35, 65, 90),
        2,
        cv2.LINE_AA
    )

    explanations = [
        "Red dots      = Microaneurysms",
        "Violet areas  = Haemorrhage",
        "Yellow areas  = Hard exudates",
        "Cyan areas    = Soft exudates"
    ]

    yy = box_y + 62

    for text in explanations:

        cv2.putText(
            panel,
            text,
            (32, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (60, 60, 60),
            1,
            cv2.LINE_AA
        )

        yy += 25

    cv2.putText(
        panel,
        "This overlay shows locations",
        (32, yy + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (90, 90, 90),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        panel,
        "identified by the AI model.",
        (32, yy + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (90, 90, 90),
        1,
        cv2.LINE_AA
    )

    return panel


# ============================================================
# FINAL OVERLAY WITH PANEL
# ============================================================

def create_overlay(
    image,
    processed_masks,
    summary
):
    """
    Create final organized TRINAY output.
    """

    retina_overlay, display_masks = (
        create_retina_overlay(
            image,
            processed_masks
        )
    )

    # Add title above retina
    retina_overlay = add_title(
        retina_overlay
    )

    # Create panel
    panel = create_information_panel(
        summary,
        panel_width=360,
        panel_height=retina_overlay.shape[0]
    )

    # Match panel height
    if panel.shape[0] != retina_overlay.shape[0]:

        panel = cv2.resize(
            panel,
            (
                panel.shape[1],
                retina_overlay.shape[0]
            ),
            interpolation=cv2.INTER_AREA
        )

    # Combine
    final = np.hstack(
        [
            retina_overlay,
            panel
        ]
    )

    return final


# ============================================================
# SIDE-BY-SIDE COMPARISON
# ============================================================

def create_comparison(
    original,
    overlay
):
    """
    Create a simple original-vs-TRINAY comparison.
    """

    original = add_title(
        original
    )

    # Make heights equal
    target_height = max(
        original.shape[0],
        overlay.shape[0]
    )

    def pad_height(
        image,
        target
    ):

        h, w = image.shape[:2]

        if h >= target:
            return image

        canvas = np.full(
            (
                target,
                w,
                3
            ),
            248,
            dtype=np.uint8
        )

        canvas[:h] = image

        return canvas

    original = pad_height(
        original,
        target_height
    )

    overlay = pad_height(
        overlay,
        target_height
    )

    separator = np.full(
        (
            target_height,
            12,
            3
        ),
        225,
        dtype=np.uint8
    )

    comparison = np.hstack(
        [
            original,
            separator,
            overlay
        ]
    )

    return comparison