#!/usr/bin/env python3
"""Generate an A4 landscape checkerboard for camera calibration."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


SQUARE_COLUMNS = 10
SQUARE_ROWS = 7
INNER_CORNERS = (9, 6)
SQUARE_SIZE_M = 0.020

PAGE_WIDTH_MM = 297
PAGE_HEIGHT_MM = 210
PIXELS_PER_MM = 10
DPI = PIXELS_PER_MM * 25.4

OUTPUT_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_STEM = "checkerboard_10x7_a4_landscape_20mm"


def main() -> None:
    # ── Render the exact 200 x 140 mm checkerboard ─────────────────────
    square_px = int(SQUARE_SIZE_M * 1000 * PIXELS_PER_MM)
    active_width_px = SQUARE_COLUMNS * square_px
    active_height_px = SQUARE_ROWS * square_px
    checkerboard = Image.new(
        "L",
        (active_width_px, active_height_px),
        255,
    )
    draw = ImageDraw.Draw(checkerboard)
    for row in range(SQUARE_ROWS):
        for column in range(SQUARE_COLUMNS):
            if (row + column) % 2 == 0:
                left = column * square_px
                top = row * square_px
                draw.rectangle(
                    (
                        left,
                        top,
                        left + square_px - 1,
                        top + square_px - 1,
                    ),
                    fill=0,
                )

    # ── Centre it on an exact A4 landscape page ────────────────────────
    page_width_px = PAGE_WIDTH_MM * PIXELS_PER_MM
    page_height_px = PAGE_HEIGHT_MM * PIXELS_PER_MM
    left_px = (page_width_px - active_width_px) // 2
    top_px = (page_height_px - active_height_px) // 2
    page = Image.new("L", (page_width_px, page_height_px), 255)
    page.paste(checkerboard, (left_px, top_px))

    png_path = OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.pdf"
    page.save(png_path, dpi=(DPI, DPI), optimize=True)
    page.convert("1", dither=Image.Dither.NONE).save(
        pdf_path,
        "PDF",
        resolution=DPI,
    )

    # ── Verify all OpenCV inner corners before reporting success ───────
    detected, corners = cv2.findChessboardCornersSB(
        np.asarray(page),
        INNER_CORNERS,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    detected_corners = 0 if corners is None else len(corners)
    if not detected or detected_corners != 54:
        raise RuntimeError(
            "Generated checkerboard validation failed: "
            f"{detected_corners} inner corners"
        )

    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"Validated {detected_corners} OpenCV inner corners.")
    print("Print the PDF at 100% / Actual size; disable Fit or Scale to page.")


if __name__ == "__main__":
    main()
