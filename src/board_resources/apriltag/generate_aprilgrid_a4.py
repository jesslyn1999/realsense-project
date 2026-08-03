#!/usr/bin/env python3
"""Generate the recommended A4 landscape tag36h11 AprilGrid."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


TAG_COLUMNS = 6
TAG_ROWS = 4
TAG_SIZE_M = 0.025
TAG_GAP_M = 0.0075
TAG_SPACING_RATIO = 0.30
TAG_BORDER_BITS = 1

PAGE_WIDTH_MM = 297
PAGE_HEIGHT_MM = 210
PIXELS_PER_MM = 20
DPI = PIXELS_PER_MM * 25.4

OUTPUT_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_STEM = "aprilgrid_6x4_tag36h11_a4_25mm_7p5mm"


def main() -> None:
    # ── Render the exact 187.5 x 122.5 mm AprilGrid ────────────────────
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    tag_px = int(TAG_SIZE_M * 1000 * PIXELS_PER_MM)
    gap_px = int(TAG_GAP_M * 1000 * PIXELS_PER_MM)
    grid_width_px = TAG_COLUMNS * tag_px + (TAG_COLUMNS - 1) * gap_px
    grid_height_px = TAG_ROWS * tag_px + (TAG_ROWS - 1) * gap_px

    # ── Centre it on an exact A4 landscape page ────────────────────────
    page_width_px = PAGE_WIDTH_MM * PIXELS_PER_MM
    page_height_px = PAGE_HEIGHT_MM * PIXELS_PER_MM
    left_px = (page_width_px - grid_width_px) // 2
    top_px = (page_height_px - grid_height_px) // 2
    page = np.full((page_height_px, page_width_px), 255, dtype=np.uint8)

    for row in range(TAG_ROWS):
        for column in range(TAG_COLUMNS):
            marker_id = row * TAG_COLUMNS + column
            marker = cv2.aruco.generateImageMarker(
                dictionary,
                marker_id,
                tag_px,
                borderBits=TAG_BORDER_BITS,
            )
            left = left_px + column * (tag_px + gap_px)
            top = top_px + row * (tag_px + gap_px)
            page[top : top + tag_px, left : left + tag_px] = marker

    image = Image.fromarray(page)
    png_path = OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.pdf"
    image.save(png_path, dpi=(DPI, DPI), optimize=True)
    image.convert("1", dither=Image.Dither.NONE).save(
        pdf_path,
        "PDF",
        resolution=DPI,
    )

    # ── Verify all tag36h11 IDs before reporting success ───────────────
    marker_corners, marker_ids, rejected = cv2.aruco.ArucoDetector(
        dictionary
    ).detectMarkers(page)
    detected_ids = (
        []
        if marker_ids is None
        else sorted(int(value) for value in marker_ids.reshape(-1))
    )
    expected_ids = list(range(TAG_COLUMNS * TAG_ROWS))
    if detected_ids != expected_ids:
        raise RuntimeError(
            "Generated AprilGrid validation failed: "
            f"detected IDs {detected_ids}"
        )

    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(
        f"Validated {len(detected_ids)} tag36h11 markers with IDs "
        f"{detected_ids[0]}-{detected_ids[-1]}."
    )
    print(
        f"Kalibr tagSize={TAG_SIZE_M:.3f} m, "
        f"tagSpacing={TAG_SPACING_RATIO:.2f}."
    )
    print("Print the PDF at 100% / Actual size; disable Fit or Scale to page.")


if __name__ == "__main__":
    main()
