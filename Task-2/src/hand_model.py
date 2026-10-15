"""Shared HandLandmarker setup, so every notebook loads the model with the
same options and gets the same (corrected) handedness convention.

Note: pinned to mediapipe==0.10.35 — version 1.0.1 crashes on Apple Silicon
Macs when loading HandLandmarker (see README.md for details).
"""
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

DEFAULT_MODEL_PATH = "../models/hand_landmarker.task"

# MediaPipe's handedness output is calibrated for a non-mirrored camera feed.
# Since every notebook here flips the frame for natural (mirror-style)
# display, MediaPipe's "Left"/"Right" ends up inverted relative to the
# player's real hand — this swap corrects it (verified empirically).
HANDEDNESS_SWAP = {"Left": "Right", "Right": "Left"}


def load_hand_landmarker(
    model_path=DEFAULT_MODEL_PATH,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
):
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def corrected_handedness(category_name):
    """Swap MediaPipe's raw 'Left'/'Right' to match the player's real hand
    (see HANDEDNESS_SWAP note above)."""
    return HANDEDNESS_SWAP.get(category_name, category_name)
