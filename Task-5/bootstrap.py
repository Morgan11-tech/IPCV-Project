"""
Task 5 - import this FIRST. Makes every task folder importable and points
the task modules at shared paths and the game's player colours.

Folder names contain '-' so they cannot be Python packages; adding them to
sys.path lets the game import each task's modules by name.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for folder in ("Task-1", "Task-2", "Task-3", "Task-4", "Task-5"):
    path = os.path.join(ROOT, folder)
    if path not in sys.path:
        sys.path.insert(0, path)

import config                       # noqa: E402  (Task 1)
import task3_player_association     # noqa: E402  (Task 3)
import theme                        # noqa: E402  (Task 5)

# Task 1's config uses paths relative to Task-1/; make them absolute so the
# game can be started from anywhere. Registered faces live in <root>/player_data.
config.DETECTOR_MODEL = os.path.join(ROOT, "Task-1", "models", "face_detection_yunet_2023mar.onnx")
config.RECOGNIZER_MODEL = os.path.join(ROOT, "Task-1", "models", "face_recognition_sface_2021dec.onnx")
config.PLAYER_DATA_DIR = os.path.join(ROOT, "player_data")
POSE_MODEL = os.path.join(ROOT, "Task-3", "models", "pose_landmarker_lite.task")
ASSETS = os.path.join(ROOT, "Task-5", "assets")

# One colour per player everywhere (Task 1 face ring/tag, Task 3 debug view).
config.PLAYER_NAMES = dict(theme.PLAYER_NAMES)
config.PLAYER_COLORS = dict(theme.PLAYER_COLORS)
task3_player_association.PLAYER_DEBUG_COLORS = dict(theme.PLAYER_COLORS)
