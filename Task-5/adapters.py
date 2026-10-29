"""
Task 5 - glue between the task modules.

Each function turns one task's output into the next task's input, so no
task module has to know about the others:

    Task 2 hands (normalised)  -> Task 3 hands (pixels)        hands_for_task3()
    Task 1 + Task 2 + Task 3   -> Task 4 PlayerInput per player player_inputs()
    (testing alone)            -> a fake, always-ready Player 2 fake_player_input()

Word labels: Task 2's model returns capitalised words ("Hello") while Task 4's
level lists are lower case ("hello"). Everything handed to Task 4 is
lower-cased here, otherwise word_probs.get(target) would always be 0.
"""
import numpy as np

from task4_game import PlayerInput, synthetic_hand

IDLE_LABEL = "idle"     # Task 2's "nothing is being signed" class


def _word(label):
    """Task 2 label -> Task 4 word: lower case, None for Idle / no guess."""
    if not label:
        return None
    word = str(label).strip().lower()
    return None if word == IDLE_LABEL else word


def hand_points_px(hand, W, H):
    """Task 2 hand (normalised x, y, z) -> (21, 3) pixel array for Task 4."""
    pts = np.asarray(hand["points"], dtype=np.float32).reshape(21, 3).copy()
    pts[:, 0] *= W
    pts[:, 1] *= H
    pts[:, 2] *= W          # MediaPipe z uses roughly the same scale as x
    return pts


def hands_for_task3(t2_out, W, H):
    """Task 2 output -> the hand list Task 3's PlayerAssociator expects.

    The list keeps Task 2's order, so index i here is hand i in Task 2's
    output and in rec.signs. Task 2's handedness is already the person's real
    hand, so Task 3 can use it instead of guessing from the face position.
    """
    hands = []
    for hand in t2_out["hands"]:
        pts = hand_points_px(hand, W, H)
        hands.append({
            "center": [float(pts[:, 0].mean()), float(pts[:, 1].mean())],
            "sign": None,
            "handedness": hand.get("handedness"),
            "points": pts[:, :2].tolist(),
        })
    return hands


def word_probs(sign):
    """Task 2 sign info -> {word: probability} for Task 4 (lower-case keys).

    Task 2 currently exposes only its top 3 guesses, so a target outside the
    top 3 counts as probability 0.
    """
    probs = {}
    for label, p in sign.get("top3") or []:
        word = _word(label)
        if word is not None:
            probs[word] = float(p)
    return probs


def _best_sign(indices, signs):
    """Of one player's hands, the sign that matters this frame:
    a confirmed sign first, otherwise the most confident guess."""
    best = None
    for i in indices:
        s = signs[i]
        if s.get("confirmed"):
            return s
        if s.get("guess") and (best is None or s["confidence"] > best["confidence"]):
            best = s
    return best


def player_inputs(players, owners, t2_out, signs, W, H, player_ids=(1, 2)):
    """Build Task 4's PlayerInput for every player.

    players: Task 1 {pid: PlayerState} (face tracking)
    owners:  Task 3 PlayerAssociator.current_owners {hand index: pid}
    t2_out, signs: Task 2 rec.update() output and rec.signs (same order)
    """
    inputs = {}
    for pid in player_ids:
        idx = [i for i, owner in owners.items() if owner == pid]
        face = players.get(pid)
        visible = face is not None and face.visible
        inp = PlayerInput(
            present=visible or bool(idx),
            hands=[hand_points_px(t2_out["hands"][i], W, H) for i in idx],
            face_box=tuple(face.smooth_box) if visible else None,
        )
        sign = _best_sign(idx, signs)
        if sign is not None:
            inp.locked = bool(sign.get("confirmed"))
            inp.word = _word(sign["label"] if inp.locked else sign["guess"])
            inp.conf = float(sign.get("confidence", 0.0))
            inp.word_probs = word_probs(sign)
        inputs[pid] = inp
    return inputs


def fake_player_input(W, H):
    """A stand-in Player 2 for testing alone: always present, holds a still
    open palm (so the game can start) and never signs."""
    face = (W * 0.78, H * 0.22, 110.0, 130.0)
    palm = synthetic_hand((W * 0.85, H * 0.55), 110, open_palm=True)
    return PlayerInput(present=True, hands=[palm], face_box=face)
