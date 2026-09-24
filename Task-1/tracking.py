"""
Player tracking state, EMA smoothing, short-occlusion handling and the small
game-state interface used by other modules (score, feedback).

A player's status is one of:
    SEARCHING  never seen yet
    TRACKED    matched to a detected face in this frame
    HELD       not detected, but missing for <= MAX_MISSING_FRAMES:
               effect stays at the last smoothed position
    LOST       missing for longer: effect hidden until the player is
               recognised again
"""
import math
import time
from dataclasses import dataclass

import config


@dataclass
class PlayerState:
    player_id: int
    name: str
    color: tuple
    # --- tracking state ---
    raw_box: tuple = None        # (x, y, w, h) straight from the detector
    smooth_box: tuple = None     # (x, y, w, h) after EMA; used for drawing
    landmarks: object = None     # 5x2 array from the last detection
    roll: float = 0.0            # smoothed head roll in degrees
    similarity: float = 0.0      # cosine similarity of the last match
    source: str = ""             # "recognized" (embedding) or "overlap" (IoU)
    status: str = "SEARCHING"
    last_seen_frame: int = -1
    missing_frames: int = 0
    # --- game state (written by the game module later) ---
    score: int = 0
    feedback_text: str = ""
    feedback_until: float = 0.0

    @property
    def visible(self):
        return self.status in ("TRACKED", "HELD")

    @property
    def center(self):
        x, y, w, h = self.smooth_box
        return (x + w / 2, y + h / 2)

    @property
    def face_width(self):
        return self.smooth_box[2]

    @property
    def face_height(self):
        return self.smooth_box[3]


def create_players():
    return {pid: PlayerState(pid, config.PLAYER_NAMES[pid], config.PLAYER_COLORS[pid])
            for pid in (1, 2)}


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def smooth_value(previous, current, alpha=config.EMA_ALPHA):
    """
    Exponential Moving Average:
        smoothed = alpha * current + (1 - alpha) * previous_smoothed

    The detector's box moves by a few pixels every frame even for a still head
    (sensor noise, compression, the CNN re-estimating the box from scratch each
    frame). This noise is high-frequency; true head motion is mostly slower.
    The EMA is a first-order low-pass filter: it keeps the slow part and damps
    the fast part. Unrolled, each past frame's weight decays by (1 - alpha),
    so older measurements fade out exponentially.
    """
    return alpha * current + (1 - alpha) * previous


def smooth_bbox(previous_box, current_box, alpha=config.EMA_ALPHA):
    """Apply the EMA independently to x, y, width and height."""
    return tuple(smooth_value(p, c, alpha) for p, c in zip(previous_box, current_box))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def box_iou(a, b):
    """Intersection-over-Union of two (x, y, w, h) boxes: 0 = disjoint, 1 = identical."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = ix * iy
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def compute_roll_angle(landmarks):
    """
    Head roll (in-plane tilt) from the two eye landmarks:
        angle = atan2(eye2_y - eye1_y, eye2_x - eye1_x)
    with eye1 the eye further left in the image. Image y points down, so a
    positive angle is a clockwise tilt on screen, which is the same convention
    cv2.ellipse uses for its rotation angle.
    """
    eye_a, eye_b = landmarks[0], landmarks[1]
    if eye_a[0] > eye_b[0]:
        eye_a, eye_b = eye_b, eye_a
    return math.degrees(math.atan2(eye_b[1] - eye_a[1], eye_b[0] - eye_a[0]))


# ---------------------------------------------------------------------------
# Track updates
# ---------------------------------------------------------------------------
def update_player_track(player, face, similarity, source, frame_idx):
    """A detection was matched to this player: update raw and smoothed state."""
    box = face["box"]
    roll = compute_roll_angle(face["landmarks"])

    # Decide whether to continue the EMA or restart it. Restart when there is
    # no valid previous estimate, or when the face is far from where the
    # effect currently is (re-acquisition after LOST or an identity
    # correction): blending those would make the effect slide across the
    # screen, which would look like the effect moving between players.
    restart = player.smooth_box is None or player.status == "LOST"
    if not restart:
        cx, cy = player.center
        jump = math.hypot(box[0] + box[2] / 2 - cx, box[1] + box[3] / 2 - cy)
        restart = jump > config.SMOOTHING_RESET_DISTANCE * player.face_width

    if restart:
        player.smooth_box = box
        player.roll = roll
    else:
        player.smooth_box = smooth_bbox(player.smooth_box, box)
        player.roll = smooth_value(player.roll, roll)

    player.raw_box = box
    player.landmarks = face["landmarks"]
    player.similarity = similarity
    player.source = source
    player.status = "TRACKED"
    player.last_seen_frame = frame_idx
    player.missing_frames = 0


def mark_player_missing(player):
    """
    No detection for this player in this frame. Short gaps (blink of the
    detector, hand in front of face) keep the last smoothed box so the effect
    does not flicker; longer gaps switch the player to LOST.
    """
    if player.status in ("TRACKED", "HELD"):
        player.missing_frames += 1
        player.status = "HELD" if player.missing_frames <= config.MAX_MISSING_FRAMES else "LOST"
    elif player.status == "LOST":
        player.missing_frames += 1


def update_all_tracks(players, faces, assignments, similarities, frame_idx):
    """
    Update both players from this frame's detections.

    Step 1: players assigned by face recognition (embedding similarity) take
            their face. This is what keeps identities correct when players
            cross or leave and re-enter: identity comes from the face, not
            from the position.
    Step 2: a player that was NOT recognised this frame but is still being
            tracked may keep a remaining face that overlaps its last box
            (IoU >= TRACK_IOU_THRESHOLD). This bridges frames where the
            similarity drops (head turned, partial occlusion, or frames on
            which recognition is skipped for speed).
    Step 3: players without a face are marked missing.

    Returns the indices of faces that belong to nobody (UNKNOWN).
    """
    matched = {}
    used = set()
    for pid, (f, sim) in assignments.items():
        matched[pid] = (f, sim, "recognized")
        used.add(f)

    for pid, player in players.items():
        if pid in matched or not player.visible:
            continue
        best_f, best_iou = None, 0.0
        for f, face in enumerate(faces):
            if f in used:
                continue
            iou = box_iou(player.raw_box, face["box"])
            if iou > best_iou:
                best_f, best_iou = f, iou
        if best_f is not None and best_iou >= config.TRACK_IOU_THRESHOLD:
            sim = float(similarities[best_f, pid - 1]) if similarities is not None else player.similarity
            matched[pid] = (best_f, sim, "overlap")
            used.add(best_f)

    for pid, player in players.items():
        if pid in matched:
            f, sim, source = matched[pid]
            update_player_track(player, faces[f], sim, source, frame_idx)
        else:
            mark_player_missing(player)

    return [f for f in range(len(faces)) if f not in used]


# ---------------------------------------------------------------------------
# Game-state interface (called by the game module / Task 4 later)
# ---------------------------------------------------------------------------
def show_feedback(player, text, duration=config.FEEDBACK_DURATION):
    """Show a short text popup above this player's face."""
    player.feedback_text = text
    player.feedback_until = time.time() + duration


def award_point(player, text="+1 CORRECT!"):
    """Give the player a point and show feedback attached to their face."""
    player.score += 1
    show_feedback(player, text)
