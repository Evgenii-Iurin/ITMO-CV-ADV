from __future__ import annotations
import sys
import os
import argparse
from pathlib import Path
import platform

import cv2
import torch
import numpy as np
import torchvision.transforms as T
from src.keypoint_detection.custom_model import HandKeypointModel

from src.keypoint_detection.gest_checking_updated import (
    is_index_finger_only_up_mediapipe,
    is_open_palm,
    is_index_finger_only_up_model,
)

DRAW_COLOR = (0, 0, 255)  # Red (BGR)
THICKNESS = 5
DEV = "/dev/video0"


def load_background(path: Path | None, frame_shape: tuple[int, int, int]) -> np.ndarray:
    if path and path.exists():
        img = cv2.imread(str(path))
        if img is not None:
            return cv2.resize(img, (frame_shape[1], frame_shape[0]))
        print(f"Failed to read {path}; using blank canvas.")
    return np.full(frame_shape, 255, dtype=np.uint8)


def preprocess_image(frame: np.ndarray) -> torch.Tensor:
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform(frame).unsqueeze(0)  # [1, 3, 224, 224]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("background", nargs="?", default=None)
    parser.add_argument(
        "--use-mymodel", action="store_true", help="Use custom PyTorch model"
    )
    args = parser.parse_args()

    os_name = platform.system()
    if os_name == "Windows":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam.")

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read from webcam.")

    background = load_background(
        Path(args.background) if args.background else None, first_frame.shape
    )
    canvas = background.copy().astype(np.uint8)

    prev_point: tuple[int, int] | None = None
    use_model = args.use_mymodel

    # Load PyTorch model if needed
    if use_model:
        model_path = "src/keypoint_detection/assets/resnet_based_model.pt"
        model = torch.load(
            model_path,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
            weights_only=False,
        )
        model.eval()

    else:
        import mediapipe as mp

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
        )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame captured. Exiting...")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            drawing_mode = False

            if use_model:
                orig_frame = frame.copy()
                input_tensor = preprocess_image(orig_frame)
                device = next(model.parameters()).device
                input_tensor = input_tensor.to(device)

                with torch.no_grad():
                    preds = model(input_tensor)

                pred_kpts = preds["keypoints"][0].cpu().numpy()  # [21, 2]

                # ----- DRAW KEYPOINTS -----
                for x_norm, y_norm in pred_kpts:
                    x_px, y_px = int(x_norm * w), int(y_norm * h)
                    cv2.circle(frame, (x_px, y_px), 3, (0, 255, 0), -1)  # green dot

                # ----- GESTURE CHECK -----
                if is_index_finger_only_up_model(pred_kpts):
                    drawing_mode = True

                elif is_open_palm(
                    [type("Dummy", (), {"y": y, "x": x}) for x, y in pred_kpts]
                ):
                    prev_point = None

                # ----- DRAWING -----
                if drawing_mode:
                    px, py = int(pred_kpts[8][0] * w), int(pred_kpts[8][1] * h)
                    current_point = (px, py)
                    if prev_point is not None:
                        cv2.line(
                            canvas, prev_point, current_point, DRAW_COLOR, THICKNESS
                        )
                    prev_point = current_point
                else:
                    prev_point = None

            else:
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0].landmark
                    if is_index_finger_only_up_mediapipe(hand_landmarks):
                        drawing_mode = True
                    elif is_open_palm(hand_landmarks):
                        prev_point = None

                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                    )

                    if drawing_mode:
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

            # Composite result
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
        if not use_model:
            hands.close()


if __name__ == "__main__":
    main()
