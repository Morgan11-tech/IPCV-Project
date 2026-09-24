"""
All drawing: the face-attached effect, the player tag / score (game-state
effect), the "+1 CORRECT!" feedback, UNKNOWN faces, the HUD and debug info.

Every face effect is computed from the player's SMOOTHED box, so position,
size and rotation follow the face without the detector's jitter:
    centre  = centre of the smoothed box          -> follows movement
    size    = proportional to smoothed w and h    -> follows distance
    angle   = smoothed head roll from the eyes    -> follows head tilt
"""
import math
import time

import cv2
import numpy as np

import config

FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_text(frame, text, org, scale, color, thickness=2):
    """Text with a dark outline so it stays readable on any background."""
    org = (int(org[0]), int(org[1]))
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def put_text_centered(frame, text, center_x, y, scale, color, thickness=2):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    put_text(frame, text, (center_x - tw / 2, y), scale, color, thickness)


def text_scale_for(face_width):
    """Labels grow and shrink with the face so they stay proportional."""
    return float(np.clip(face_width / 180.0, 0.45, 1.2))


def rotate_point(cx, cy, dx, dy, angle_deg):
    """Rotate offset (dx, dy) around (cx, cy) by angle (image coords, y down)."""
    a = math.radians(angle_deg)
    return (cx + dx * math.cos(a) - dy * math.sin(a),
            cy + dx * math.sin(a) + dy * math.cos(a))


# ---------------------------------------------------------------------------
# Effect 1: the face-attached effect
# ---------------------------------------------------------------------------
def draw_face_effect(frame, player):
    """
    A coloured halo (ellipse) around the face plus a small marker on top of
    the head. Both are rotated by the head roll angle, so the effect tilts
    with the head. The halo's axes scale with the face box.
    """
    if not player.visible:
        return
    cx, cy = player.center
    w, h = player.face_width, player.face_height
    axes = (int(w * 0.62), int(h * 0.70))
    thickness = max(2, int(w * 0.04))
    if player.status == "HELD":
        thickness = max(1, thickness // 2)   # thinner while the face is not detected

    center = (int(cx), int(cy))
    cv2.ellipse(frame, center, axes, player.roll, 0, 360, player.color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, center, (axes[0] + thickness * 2, axes[1] + thickness * 2),
                player.roll, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)

    # Marker on top of the head: a triangle pointing up along the face's own
    # vertical axis (rotated by roll).
    top = axes[1] + thickness * 3
    size = w * 0.12
    points = [rotate_point(cx, cy, 0, -top - size, player.roll),
              rotate_point(cx, cy, -size * 0.7, -top, player.roll),
              rotate_point(cx, cy, size * 0.7, -top, player.roll)]
    cv2.fillPoly(frame, [np.array(points, dtype=np.int32)], player.color, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Effect 2: game state attached to the player (name + score) and feedback
# ---------------------------------------------------------------------------
def draw_player_tag(frame, player):
    """Name and score shown under the chin, following and scaling with the face."""
    if not player.visible:
        return
    cx, _ = player.center
    x, y, w, h = player.smooth_box
    scale = text_scale_for(w)
    label = f"{player.name} | {player.score} pt"
    (tw, th), _ = cv2.getTextSize(label, FONT, scale, 2)
    top = int(y + h * 1.25)
    x0, x1 = int(cx - tw / 2 - 8), int(cx + tw / 2 + 8)
    cv2.rectangle(frame, (x0, top), (x1, top + th + 14), player.color, -1)
    cv2.putText(frame, label, (x0 + 8, top + th + 6), FONT, scale, (255, 255, 255), 2, cv2.LINE_AA)


def draw_feedback(frame, player, now=None):
    """
    Temporary "+1 CORRECT!" popup above the player's head. It floats upward
    over its lifetime so it reads as an event, not a static label.
    """
    now = time.time() if now is None else now
    if not player.visible or now > player.feedback_until:
        return
    progress = 1.0 - (player.feedback_until - now) / config.FEEDBACK_DURATION
    cx, _ = player.center
    x, y, w, h = player.smooth_box
    scale = text_scale_for(w) * 1.3
    rise = progress * h * 0.5
    put_text_centered(frame, player.feedback_text, cx, y - h * 0.45 - rise, scale, (60, 220, 60), 3)


def draw_unknown_face(frame, face, similarities_row=None, debug=False):
    """Faces that match neither player: grey box and UNKNOWN label."""
    x, y, w, h = [int(v) for v in face["box"]]
    cv2.rectangle(frame, (x, y), (x + w, y + h), config.UNKNOWN_COLOR, 2)
    put_text(frame, "UNKNOWN", (x, y - 8), text_scale_for(w), config.UNKNOWN_COLOR)
    if debug and similarities_row is not None:
        put_text(frame, f"P1 {similarities_row[0]:.2f}  P2 {similarities_row[1]:.2f}",
                 (x, y + h + 20), 0.5, config.UNKNOWN_COLOR, 1)


def draw_hud(frame, players):
    """Fixed score panels in the top corners (status of each player)."""
    frame_w = frame.shape[1]
    for pid, player in players.items():
        status = {"TRACKED": "READY", "HELD": "READY", "LOST": "NOT VISIBLE",
                  "SEARCHING": "SEARCHING"}[player.status]
        x0 = 15 if pid == 1 else frame_w - 235
        cv2.rectangle(frame, (x0, 15), (x0 + 220, 110), (30, 30, 30), -1)
        cv2.rectangle(frame, (x0, 15), (x0 + 220, 110), player.color, 3)
        put_text(frame, player.name, (x0 + 12, 45), 0.8, player.color)
        put_text(frame, f"Score: {player.score}", (x0 + 12, 72), 0.65, (255, 255, 255))
        put_text(frame, status, (x0 + 12, 98), 0.6,
                 (60, 220, 60) if player.visible else (0, 170, 255))


def draw_debug(frame, players, fps, timings, recording=False):
    """Raw vs. smoothed boxes, similarity, track source, roll, FPS and stage timing."""
    for player in players.values():
        if not player.visible:
            continue
        if player.status == "TRACKED":
            x, y, w, h = [int(v) for v in player.raw_box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)   # raw detector box
            for lx, ly in player.landmarks:
                cv2.circle(frame, (int(lx), int(ly)), 2, (0, 255, 255), -1)
        x, y, w, h = [int(v) for v in player.smooth_box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), player.color, 1)          # smoothed box
        info = [f"sim {player.similarity:.2f} ({player.source})",
                f"{player.status} miss {player.missing_frames}  roll {player.roll:+.0f}"]
        for i, line in enumerate(info):
            put_text(frame, line, (x + w + 8, y + 18 + 20 * i), 0.5, player.color, 1)

    lines = [f"FPS {fps:5.1f}   frame {timings.get('total', 0):5.1f} ms",
             f"detect {timings.get('detect', 0):4.1f}  recog {timings.get('recognize', 0):4.1f}  "
             f"track+draw {timings.get('track_draw', 0):4.1f} ms"]
    if recording:
        lines.append("REC evaluation trial (E to stop)")
    h = frame.shape[0]
    for i, line in enumerate(lines):
        put_text(frame, line, (15, h - 20 - 25 * (len(lines) - 1 - i)), 0.6,
                 (0, 0, 255) if line.startswith("REC") else (255, 255, 255), 1)


# ---------------------------------------------------------------------------
# Screens for the registration / preparation phases
# ---------------------------------------------------------------------------
def draw_text_panel(frame, lines, color=(255, 255, 255)):
    """Dark panel at the top of the frame with instruction lines."""
    height = 30 + 34 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, dst=frame)
    for i, line in enumerate(lines):
        put_text(frame, line, (25, 42 + 34 * i), 0.9 if i == 0 else 0.7,
                 color if i == 0 else (255, 255, 255), 2 if i == 0 else 1)
