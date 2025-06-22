from __future__ import annotations

import sys
from pathlib import Path
import platform

import cv2
import mediapipe as mp
import numpy as np

from src.hw_7.gest_checking import is_index_finger_only_up, is_open_palm

DEV = "/dev/video0"
DRAW_COLOR = (0, 0, 255)  # red in BGR
THICKNESS = 5

# ------------------------
# Helper functions
# ------------------------


def load_background(path: Path | None, frame_shape: tuple[int, int, int]) -> np.ndarray:
    """Return an image to be used as the drawing canvas background.

    If *path* is None or invalid, create a plain white canvas matching *frame_shape*.
    """
    # pylint: disable=no-member
    if path and path.exists():
        img = cv2.imread(str(path))
        if img is None:
            print(f"Failed to read {path}; falling back to white canvas.")
        else:
            return cv2.resize(img, (frame_shape[1], frame_shape[0]))
    return np.full(frame_shape, 255, dtype=np.uint8)


def main():
    """
    Draw on a virtual canvas with your index finger.

    Usage:
        python virtual_whiteboard.py          # blank white canvas
        python virtual_whiteboard.py path/to/background.jpg

    Controls:
        - Raise only your index finger to draw.
        - Open your palm (all fingers up) to pause / reset drawing point.
        - Press 'c' to clear the entire canvas.
        - Press 'q' or ESC to quit.
    """
    # Set up camera
    # pylint: disable=no-member
    os_name = platform.system()

    if os_name == "Windows":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    else:  # Linux / WSL
        cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot access webcam. Make sure a camera is connected and not used by another app."
        )

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read from webcam.")

    # Load background image
    bg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    background = load_background(bg_path, first_frame.shape)
    canvas = background.copy().astype(np.uint8)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3,
    )

    prev_point: tuple[int, int] | None = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame captured. Exiting…")
                break

            # mirror for natural interaction
            frame = cv2.flip(frame, 1)
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            drawing_mode = False
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0].landmark
                if is_index_finger_only_up(hand_landmarks):
                    drawing_mode = True
                elif is_open_palm(hand_landmarks):
                    prev_point = None  # reset on open palm

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )

                if drawing_mode:
                    h, w, _ = frame.shape
                    x_px = int(hand_landmarks[8].x * w)
                    y_px = int(hand_landmarks[8].y * h)
                    current_point = (x_px, y_px)
                    if prev_point is not None:
                        cv2.line(
                            canvas, prev_point, current_point, DRAW_COLOR, THICKNESS
                        )
                    prev_point = current_point
                else:
                    prev_point = None
            else:
                prev_point = None

            gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 250, 255, cv2.THRESH_BINARY_INV)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            background_part = cv2.bitwise_and(frame, cv2.bitwise_not(mask_bgr))
            foreground_part = cv2.bitwise_and(canvas, mask_bgr)
            display = cv2.add(background_part, foreground_part)

            cv2.imshow("Virtual Whiteboard", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("c"):
                canvas[:] = background

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
