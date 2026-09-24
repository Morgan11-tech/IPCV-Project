"""
Camera helpers and player registration.

Registration = collect NUM_REGISTRATION_SAMPLES embeddings of one player.
For each SPACE press:
    detect face -> check it is a valid sample -> align + embed -> store
"""
import os
import time

import cv2
import numpy as np

import config
from effects import draw_text_panel
from recognition import cosine_similarity, detect_faces, get_face_embedding


# ---------------------------------------------------------------------------
# Camera helpers (shared with main.py)
# ---------------------------------------------------------------------------
def open_camera(index=config.CAMERA_INDEX):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. On macOS, allow camera access "
                           "for your terminal/IDE in System Settings > Privacy & Security.")
    return cap


def read_frame(cap):
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.flip(frame, 1) if config.MIRROR_VIEW else frame


def show_message(cap, window, lines, seconds, color=(60, 220, 60)):
    """Show a message over the live camera image for a few seconds."""
    end = time.time() + seconds
    while time.time() < end:
        frame = read_frame(cap)
        if frame is None:
            break
        draw_text_panel(frame, lines, color)
        cv2.imshow(window, frame)
        cv2.waitKey(1)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def check_registration_face(faces):
    """
    Return None if the frame contains exactly one usable face, otherwise a
    short reason. Registering from a frame with two faces could store the
    wrong person's embedding, and blurry/small/low-confidence faces give
    embeddings that describe the image quality more than the person.
    """
    if len(faces) == 0:
        return "No face visible"
    if len(faces) > 1:
        return "More than one face visible - only this player should be in view"
    face = faces[0]
    if face["score"] < config.REGISTRATION_MIN_SCORE:
        return f"Face not clear enough (confidence {face['score']:.2f})"
    if face["box"][2] < config.REGISTRATION_MIN_FACE_WIDTH:
        return "Face too small - move closer"
    return None


def register_player(cap, window, detector, recognizer, player_id, other_reference=None):
    """
    Interactive registration of one player.

    other_reference: reference embedding of the already-registered player.
    A sample that matches it is rejected, so the same person cannot be
    registered as both players by mistake.

    Returns (embeddings array of shape (N, 128), list of aligned crops),
    or None if the user quits.
    """
    name = config.PLAYER_NAMES[player_id]
    color = config.PLAYER_COLORS[player_id]
    embeddings, crops = [], []
    message, message_until = "", 0.0

    while len(embeddings) < config.NUM_REGISTRATION_SAMPLES:
        frame = read_frame(cap)
        if frame is None:
            raise RuntimeError("Camera stopped delivering frames.")
        faces = detect_faces(detector, frame)
        problem = check_registration_face(faces)

        display = frame.copy()
        for face in faces:
            x, y, w, h = [int(v) for v in face["box"]]
            cv2.rectangle(display, (x, y), (x + w, y + h),
                          (60, 220, 60) if problem is None else (0, 0, 255), 2)
        lines = [f"REGISTER {name}",
                 "Look at the camera. Between captures, slightly change head angle,",
                 "expression and distance.",
                 f"Samples: {len(embeddings)} / {config.NUM_REGISTRATION_SAMPLES}",
                 "SPACE capture    R restart    Q quit"]
        if time.time() < message_until:
            lines.append(message)
        elif problem:
            lines.append(problem)
        draw_text_panel(display, lines, color)
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return None
        if key == ord("r"):
            embeddings, crops = [], []
            message, message_until = "Restarted registration", time.time() + 1.5
        if key == ord(" "):
            if problem is not None:
                message, message_until = f"Sample rejected: {problem}", time.time() + 1.5
                continue
            embedding = get_face_embedding(recognizer, frame, faces[0])
            if (other_reference is not None and
                    cosine_similarity(embedding, other_reference) >= config.SIMILARITY_THRESHOLD):
                message = "Sample rejected: this face matches the other registered player"
                message_until = time.time() + 2.0
                continue
            embeddings.append(embedding)
            crops.append(recognizer.alignCrop(frame, faces[0]["row"]))
            message, message_until = f"Captured sample {len(embeddings)}", time.time() + 0.8

    show_message(cap, window, [f"{name} registered successfully."], 1.5)
    return np.array(embeddings), crops


# ---------------------------------------------------------------------------
# Saving / loading (so evaluation trials can reuse one registration)
# ---------------------------------------------------------------------------
def player_dir(player_id):
    return os.path.join(config.PLAYER_DATA_DIR, f"player_{player_id}")


def save_player_data(player_id, embeddings, crops):
    """Store embeddings (.npy) and the aligned 112x112 crops (for inspection)."""
    folder = player_dir(player_id)
    os.makedirs(folder, exist_ok=True)
    for old in os.listdir(folder):
        if old.startswith("sample_"):
            os.remove(os.path.join(folder, old))
    np.save(os.path.join(folder, "embeddings.npy"), embeddings)
    for i, crop in enumerate(crops, start=1):
        cv2.imwrite(os.path.join(folder, f"sample_{i:02d}.jpg"), crop)


def load_player_data(player_id):
    path = os.path.join(player_dir(player_id), "embeddings.npy")
    return np.load(path) if os.path.exists(path) else None
