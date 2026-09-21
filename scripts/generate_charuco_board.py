"""Generates the ChArUco calibration board used for Track 2 (camera
calibration), sized to print cleanly on a single US Letter sheet at 300 DPI.

These exact parameters (squares, dictionary, physical sizes) will need to be
passed to cv2.aruco.CharucoBoard again during actual calibration later, so
keep this file as the single source of truth for them rather than
re-guessing the board's real-world dimensions from a printout.

IMPORTANT: when printing the output image, use "Actual Size" / 100% scale,
NOT "Fit to Page" - the whole point of a calibration board is that the
software knows its real-world square size in millimeters, which only holds
if it's printed at the size this script assumes.

Usage: python scripts/generate_charuco_board.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "configs" / "charuco_board_letter.png"

DPI = 300
PAGE_WIDTH_IN = 8.5
PAGE_HEIGHT_IN = 11.0

SQUARES_X = 7
SQUARES_Y = 9
SQUARE_LENGTH_MM = 24.0
MARKER_LENGTH_MM = 17.0
ARUCO_DICT = cv2.aruco.DICT_4X4_50


def mm_to_px(mm: float) -> int:
    return round(mm / 25.4 * DPI)


def main() -> None:
    page_w = round(PAGE_WIDTH_IN * DPI)
    page_h = round(PAGE_HEIGHT_IN * DPI)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH_MM / 1000,
        MARKER_LENGTH_MM / 1000,
        dictionary,
    )
    board_w_px = mm_to_px(SQUARES_X * SQUARE_LENGTH_MM)
    board_h_px = mm_to_px(SQUARES_Y * SQUARE_LENGTH_MM)
    board_img = board.generateImage((board_w_px, board_h_px), marginSize=0, borderBits=1)

    page = np.full((page_h, page_w), 255, dtype=np.uint8)

    top_margin = mm_to_px(12)
    header_h = mm_to_px(14)
    x_offset = (page_w - board_w_px) // 2
    y_offset = top_margin + header_h
    page[y_offset:y_offset + board_h_px, x_offset:x_offset + board_w_px] = board_img

    page_bgr = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(page_bgr, "running-motion-prediction calibration board",
                (x_offset, top_margin + 30), font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(page_bgr, "PRINT AT 100% / ACTUAL SIZE - do not use 'Fit to Page'",
                (x_offset, top_margin + 65), font, 0.7, (0, 0, 200), 2, cv2.LINE_AA)

    footer_lines = [
        f"ChArUco {SQUARES_X}x{SQUARES_Y}, square={SQUARE_LENGTH_MM:.0f}mm, "
        f"marker={MARKER_LENGTH_MM:.0f}mm, dict=DICT_4X4_50",
        "After printing, measure a square with a ruler to confirm it's really "
        f"{SQUARE_LENGTH_MM:.0f}mm - if not, your printer rescaled it.",
    ]
    footer_y = y_offset + board_h_px + mm_to_px(10)
    for i, line in enumerate(footer_lines):
        cv2.putText(page_bgr, line, (x_offset, footer_y + i * 28),
                    font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), page_bgr)
    print(f"Wrote {OUTPUT_PATH} ({page_w}x{page_h}px @ {DPI} DPI, "
          f"board {SQUARES_X}x{SQUARES_Y} squares @ {SQUARE_LENGTH_MM:.0f}mm each)")


if __name__ == "__main__":
    main()
