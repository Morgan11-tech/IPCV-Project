# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 23:20:00 2026

@author: ruvah
"""

"""
Task 4 — standalone webcam demo  (DEVELOPMENT HARNESS)
======================================================
This file is NOT the deliverable.  It exists so you (Task 4) can test the
game logic in task4_game.py against a real webcam without waiting for
Task 5's integration.

Task 5 should NOT import this file.  Task 5 imports task4_game directly.

What this demo does:
  * Loads Task 2's classifier and MediaPipe landmarker.
  * Runs a webcam loop that mimics what Task 5 will eventually do.
  * Feeds PlayerInput objects to task4_game.GameController.
  * Renders the effects and HUD.
  * Handles "q" to quit and "r" to restart.

What this demo does NOT do:
  * Any scene rendering (Task 5's job).
  * Real per-player identity tracking (uses a left/right split as placeholder).
  * Real Task 1 face boxes (uses hands only).

Run:
  python task4_demo.py
or in Spyder, run the whole file.
"""

import os
import sys
import time
import importlib
import importlib.util

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------
# Import the deliverable module
# ---------------------------------------------------------------------
from task4_game import (
    GameController,
    PlayerInput,
    EffectsLayer,
    WinnerDancer,
    VictoryJingle,
    draw_hud,
    anchors_from_inputs,
    detect_defending,
    landmarks_to_xyz,
    JINGLE_ENABLED,
    JINGLE_VOLUME,
)


# ---------------------------------------------------------------------
# PATHS to Task 2's assets  (edit these once, absolute paths are robust)
# ---------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)                            # .../IPCV-Project
TASK2_DIR = os.path.join(PROJECT_ROOT, "Task-2")                # adjust if named differently
TASK2_SRC_DIR = os.path.join(TASK2_DIR, "src")
TASK2_CLASSIFIER_FILE = os.path.join(TASK2_DIR, "models", "sign_classifier.joblib")
TASK2_LANDMARKER_FILE = os.path.join(TASK2_DIR, "models", "hand_landmarker.task")

T2_BUFFER_SECONDS = 1.2
T2_MOTION_THRESHOLD = 0.03
T2_MIN_SPAN_FRACTION = 0.5
T2_CONFIDENCE_THRESHOLD = 0.30
T2_CONFIRM_SECONDS = 0.2
T2_COOLDOWN_SECONDS = 1.0
T2_IDLE_LABEL = "Idle"
T2_NUM_HANDS = 4


# ---------------------------------------------------------------------
# Task 2 asset loaders (kept in the demo, not in the deliverable)
# ---------------------------------------------------------------------
def _load_task2_classifier():
    import joblib
    if not os.path.exists(TASK2_CLASSIFIER_FILE):
        raise FileNotFoundError(f"Classifier not found at {TASK2_CLASSIFIER_FILE}")
    bundle = joblib.load(TASK2_CLASSIFIER_FILE)
    return bundle["model"], bundle["target_frames"], list(bundle["classes"])


def _load_task2_modules():
    """Load features.py and hand_model.py by absolute path so folder names
    don't matter and no __init__.py is required."""
    feat_path = os.path.join(TASK2_SRC_DIR, "features.py")
    hm_path = os.path.join(TASK2_SRC_DIR, "hand_model.py")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"features.py not found at {feat_path}")
    if not os.path.exists(hm_path):
        raise FileNotFoundError(f"hand_model.py not found at {hm_path}")

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    features_mod = _load("task2_features", feat_path)
    hand_model_mod = _load("task2_hand_model", hm_path)
    landmarker = hand_model_mod.load_hand_landmarker(
        model_path=TASK2_LANDMARKER_FILE, num_hands=T2_NUM_HANDS)
    return landmarker, features_mod


# ---------------------------------------------------------------------
# Per-player sign buffer  (Task 2's notebook logic, inlined for the demo)
# ---------------------------------------------------------------------
class _SignBuffer:
    """Per-player version of Task 2's rolling buffer + motion gate + confirm."""

    def __init__(self, clf, target_frames, classes, features_mod):
        self.clf = clf
        self.target_frames = target_frames
        self.classes = classes
        self.features = features_mod
        self.buffer = []                 # list of (t, raw63, scale)
        self.streak_label = None
        self.streak_start = 0.0
        self.last_confirmed_time = 0.0
        self.last_confirmed_label = None

    def reset(self):
        self.buffer.clear()
        self.streak_label = None
        self.streak_start = 0.0
        self.last_confirmed_time = 0.0
        self.last_confirmed_label = None

    def update(self, hand_landmarks, W, H, now):
        """hand_landmarks: list of MediaPipe NormalizedLandmark lists (0, 1, or 2 hands)."""
        if hand_landmarks:
            lm = hand_landmarks[0]
            raw63, scale = self.features.extract_frame(lm, aspect=W / max(H, 1))
            self.buffer.append((now, raw63, scale))

        while self.buffer and now - self.buffer[0][0] > T2_BUFFER_SECONDS:
            self.buffer.pop(0)

        predicted, conf = None, 0.0
        if len(self.buffer) >= 5:
            times = [t for t, _, _ in self.buffer]
            feats = [f for _, f, _ in self.buffer]
            scales = [s for _, _, s in self.buffer]
            energy = self.features.motion_energy(feats)
            span = times[-1] - times[0]
            if energy >= T2_MOTION_THRESHOLD and span >= T2_BUFFER_SECONDS * T2_MIN_SPAN_FRACTION:
                seq = self.features.build_clip_feature(feats, scales, times, self.target_frames)
                if seq is not None:
                    probs = self.clf.predict_proba([seq])[0]
                    best = int(np.argmax(probs))
                    predicted = self.clf.classes_[best]
                    conf = float(probs[best])

        locked = False
        if (predicted is not None and predicted != T2_IDLE_LABEL
                and conf >= T2_CONFIDENCE_THRESHOLD):
            if predicted == self.streak_label:
                held = now - self.streak_start
            else:
                self.streak_label, self.streak_start = predicted, now
                held = 0.0
            if held >= T2_CONFIRM_SECONDS and \
               (now - self.last_confirmed_time) >= T2_COOLDOWN_SECONDS:
                self.last_confirmed_time = now
                self.last_confirmed_label = predicted
                self.streak_label = None
                self.buffer.clear()
                locked = True
        else:
            self.streak_label = None

        if locked:
            return self.last_confirmed_label, conf, True
        if predicted is not None and predicted != T2_IDLE_LABEL:
            return predicted, conf, False
        return None, 0.0, False


# ---------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------
def run_demo(camera_index=0):
    print("Task 4 demo — checking Task 2 asset paths")
    print(f"  classifier : {TASK2_CLASSIFIER_FILE}  ->  "
          f"{'OK' if os.path.exists(TASK2_CLASSIFIER_FILE) else 'MISSING'}")
    print(f"  landmarker : {TASK2_LANDMARKER_FILE}  ->  "
          f"{'OK' if os.path.exists(TASK2_LANDMARKER_FILE) else 'MISSING'}")
    print(f"  src dir    : {TASK2_SRC_DIR}  ->  "
          f"{'OK' if os.path.isdir(TASK2_SRC_DIR) else 'MISSING'}")

    clf, target_frames, classes = _load_task2_classifier()
    landmarker, features_mod = _load_task2_modules()
    print(f"[task2] loaded classifier with {len(classes)} classes, "
          f"target_frames={target_frames}")

    # MediaPipe Pose (for defend)
    try:
        pose_det = mp.solutions.pose.Pose(model_complexity=1,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)
    except Exception as e:
        print(f"[warn] MediaPipe Pose unavailable: {e}. Defend disabled.")
        pose_det = None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {camera_index}.")

    gc = GameController()
    fx = EffectsLayer()
    dancer = WinnerDancer()
    jingle = VictoryJingle(enabled=JINGLE_ENABLED, volume=JINGLE_VOLUME)
    buffers = {pid: _SignBuffer(clf, target_frames, classes, features_mod)
               for pid in gc.ids}

    t0 = time.time()
    frame_id = 0
    next_word_at = None
    fps, last = 0.0, time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]
            t_ms = int((time.time() - t0) * 1000)
            t_s = t_ms / 1000.0

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_result = landmarker.detect(mp_image)

            pose_lms = None
            if pose_det is not None:
                pose_res = pose_det.process(rgb)
                if pose_res.pose_landmarks:
                    pose_lms = [(lm.x, lm.y, getattr(lm, "z", 0.0))
                                for lm in pose_res.pose_landmarks.landmark]

            # PLACEHOLDER identity split: left half = P1, right half = P2
            hands_by_pid = {pid: [] for pid in gc.ids}
            raw_by_pid = {pid: [] for pid in gc.ids}
            for lm_list in hand_result.hand_landmarks:
                disp_x = W - lm_list[0].x * W
                pid = gc.ids[0] if disp_x < W / 2 else gc.ids[1]
                hands_by_pid[pid].append(landmarks_to_xyz(lm_list, W, H))
                raw_by_pid[pid].append(lm_list)

            defending = detect_defending(pose_lms, W, H) if pose_lms else False

            inputs = {}
            for pid in gc.ids:
                word, conf, locked = buffers[pid].update(raw_by_pid[pid], W, H, t_s)
                inputs[pid] = PlayerInput(
                    hands=hands_by_pid[pid],
                    word=word, conf=conf, locked=locked,
                    defending=defending,
                )

            events = gc.update(t_ms, inputs, frame_id)
            for e in events:
                print(e)
                if e["type"] == "game_start":
                    gc.next_word(t_ms)
                elif e["type"] in ("score_event", "clash", "round_timeout"):
                    next_word_at = t_ms + 1500
                elif e["type"] == "game_over":
                    col = (80, 255, 80) if e["winner"] == gc.ids[0] else (0, 200, 255)
                    dancer.start(t_ms, e["winner"], col)
                    jingle.play()

            if next_word_at is not None and t_ms >= next_word_at and gc.phase == "playing":
                gc.next_word(t_ms)
                next_word_at = None

            fx.on_events(events, t_ms)

            # Draw
            view = frame.copy()
            for pid in gc.ids:
                for arr in hands_by_pid[pid]:
                    for x, y, _ in arr:
                        cv2.circle(view, (int(x), int(y)), 3, (0, 255, 0), -1)

            defaults = {gc.ids[0]: (W * 0.25, H * 0.5), gc.ids[1]: (W * 0.75, H * 0.5)}
            anchors = anchors_from_inputs(inputs, defaults)
            fx.draw(view, t_ms, anchors)
            draw_hud(view, gc.snapshot(t_ms), inputs, t_ms)
            if gc.phase == "game_over":
                dancer.draw(view, t_ms)

            fps = 0.9 * fps + 0.1 / max(time.time() - last, 1e-6)
            last = time.time()
            cv2.putText(view, f"FPS {fps:.0f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Task 4 demo (q quit, r restart)", view)

            frame_id += 1
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("r"):
                gc.reset(); fx.fx.clear(); dancer.stop(); next_word_at = None
                for b in buffers.values():
                    b.reset()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            landmarker.close()
        except Exception:
            pass
        if pose_det is not None:
            pose_det.close()


if __name__ == "__main__":
    run_demo()