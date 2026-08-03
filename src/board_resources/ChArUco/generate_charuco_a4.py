#!/usr/bin/env python3
"""Generate the recommended A4 landscape ChArUco calibration board."""

from pathlib import Path

import cv2
from PIL import Image


SQUARES_X = 10
SQUARES_Y = 7
SQUARE_SIZE_M = 0.020
MARKER_SIZE_M = 0.015
MARKER_BORDER_BITS = 1

PAGE_WIDTH_MM = 297
PAGE_HEIGHT_MM = 210
PIXELS_PER_MM = 10
DPI = PIXELS_PER_MM * 25.4

OUTPUT_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_STEM = "charuco_10x7_a4_landscape_20mm_15mm"


def main() -> None:
    # ── Build the exact 200 x 140 mm active pattern ────────────────────
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_SIZE_M,
        MARKER_SIZE_M,
        dictionary,
    )
    active_width_px = int(SQUARES_X * SQUARE_SIZE_M * 1000 * PIXELS_PER_MM)
    active_height_px = int(SQUARES_Y * SQUARE_SIZE_M * 1000 * PIXELS_PER_MM)
    active_image = board.generateImage(
        (active_width_px, active_height_px),
        marginSize=0,
        borderBits=MARKER_BORDER_BITS,
    )

    # ── Centre it on an exact A4 landscape page ────────────────────────
    page_width_px = PAGE_WIDTH_MM * PIXELS_PER_MM
    page_height_px = PAGE_HEIGHT_MM * PIXELS_PER_MM
    left_px = (page_width_px - active_width_px) // 2
    top_px = (page_height_px - active_height_px) // 2
    page = Image.new("L", (page_width_px, page_height_px), 255)
    page.paste(Image.fromarray(active_image), (left_px, top_px))

    png_path = OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.pdf"
    page.save(png_path, dpi=(DPI, DPI), optimize=True)
    page.convert("1", dither=Image.Dither.NONE).save(
        pdf_path,
        "PDF",
        resolution=DPI,
    )

    # ── Verify every marker and internal corner before reporting success
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = (
        detector.detectBoard(active_image)
    )
    detected_markers = 0 if marker_ids is None else len(marker_ids)
    detected_corners = 0 if charuco_ids is None else len(charuco_ids)
    expected_ids = list(range(35))
    detected_ids = (
        []
        if marker_ids is None
        else sorted(int(value) for value in marker_ids.reshape(-1))
    )
    if detected_ids != expected_ids or detected_corners != 54:
        raise RuntimeError(
            "Generated board validation failed: "
            f"{detected_markers} markers, {detected_corners} corners"
        )

    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(
        "Validated "
        f"{detected_markers} DICT_4X4_50 markers and "
        f"{detected_corners} ChArUco corners."
    )
    print("Print the PDF at 100% / Actual size; disable Fit or Scale to page.")


if __name__ == "__main__":
    main()
