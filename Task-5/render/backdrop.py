"""
The scene behind the game: the players placed in a rendered arena, then
colour graded so the scene reacts to the game.

Every frame:  out = (mask * camera + (1 - mask) * arena) * vignette + tint
  * mask     - person segmentation (segment.py): 1 on people, 0 on the room.
               Without a mask (arena off, B key) the plain camera is used.
  * arena    - rendered environment (arena.py)
  * vignette - slightly darker overall and darker towards the edges, so the
               players and the HUD stand out from the room.
  * tint     - a faint colour cast for the current scene: none in the lobby,
               calm green at level 1, amber at level 2, red at level 3, and
               the winner's colour at game over. It fades over TINT_FADE_MS
               instead of jumping.

Both are uint8 OpenCV operations on a mask built once, about 2 ms at 1280x720.
"""
import cv2
import numpy as np

import theme

from render.arena import Arena

BRIGHTNESS = 0.88          # centre of the image
EDGE_BRIGHTNESS = 0.55     # corners
TINT_FADE_MS = 600
LEVEL_TINTS = {1: (12, 22, 4), 2: (0, 18, 30), 3: (6, 4, 38)}   # BGR, added to the image
WINNER_TINT = 0.14         # fraction of the winner's colour added at game over


def target_tint(snap):
    if snap["phase"] == "game_over":
        if snap["winner"]:
            return tuple(c * WINNER_TINT for c in theme.player_color(snap["winner"]))
        return (0, 0, 0)
    if snap["phase"] == "playing":
        return LEVEL_TINTS.get(snap["level"], (0, 0, 0))
    return (0, 0, 0)


class Backdrop:
    def __init__(self):
        self._mask, self._size = None, None
        self.arena = Arena()
        self._tint = np.zeros(3, np.float32)
        self._from, self._to, self._t0 = self._tint.copy(), self._tint.copy(), 0

    def _build_mask(self, W, H):
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xs - W / 2) / (W / 2)) ** 2 + ((ys - H / 2) / (H / 2)) ** 2) / np.sqrt(2)
        gain = BRIGHTNESS - (BRIGHTNESS - EDGE_BRIGHTNESS) * d ** 2
        self._mask = cv2.merge([(gain * 255).astype(np.uint8)] * 3)
        self._size = (W, H)

    def _current_tint(self, snap, t):
        goal = np.array(target_tint(snap), np.float32)
        if not np.allclose(goal, self._to):
            self._from, self._to, self._t0 = self._tint.copy(), goal, t
        u = min(1.0, max(0.0, (t - self._t0) / TINT_FADE_MS))
        u = u * u * (3 - 2 * u)                       # smoothstep: ease in and out
        self._tint = self._from + (self._to - self._from) * u
        return self._tint

    def compose(self, frame, person_mask, snap, t):
        """Players from the camera over the rendered arena (blend by the mask)."""
        H, W = frame.shape[:2]
        arena = self.arena.render(snap, t, W, H)
        return cv2.blendLinear(frame, arena, person_mask, 1.0 - person_mask)

    def apply(self, frame, snap, t, person_mask=None):
        """Return the scene: the composited (or plain) camera frame, graded."""
        H, W = frame.shape[:2]
        if self._size != (W, H):
            self._build_mask(W, H)
        if person_mask is not None:
            frame = self.compose(frame, person_mask, snap, t)
        out = cv2.multiply(frame, self._mask, scale=1 / 255.0)
        tint = self._current_tint(snap, t)
        if tint.max() >= 0.5:
            out = cv2.add(out, tuple(float(c) for c in tint) + (0.0,))
        return out
