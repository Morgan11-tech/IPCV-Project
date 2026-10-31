"""
Whole-screen effects: a colour flash (correct / wrong / heart lost) and a
short screen shake. Both cost nothing while idle and about 1-2 ms while active.
"""
import cv2
import numpy as np

from render.anim import progress


class ScreenFx:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self._flash = None          # (color, t0, dur, strength)
        self._shake = None          # (t0, dur, amplitude px)

    def flash(self, color, t, dur=250, strength=0.3):
        self._flash = (color, t, dur, strength)

    def shake(self, t, dur=300, amplitude=10):
        self._shake = (t, dur, amplitude)

    def clear(self):
        self._flash = self._shake = None

    def apply(self, frame, t):
        """Apply active effects to the finished frame (returns the frame to show)."""
        if self._flash is not None:
            color, t0, dur, strength = self._flash
            u = progress(t, t0, dur)
            if u >= 1:
                self._flash = None
            else:
                a = strength * (1 - u) ** 2
                solid = np.empty_like(frame)
                solid[:] = color
                cv2.addWeighted(solid, a, frame, 1 - a, 0, dst=frame)
        if self._shake is not None:
            t0, dur, amp = self._shake
            u = progress(t, t0, dur)
            if u >= 1:
                self._shake = None
            else:
                dx, dy = self.rng.uniform(-1, 1, 2) * amp * (1 - u)
                m = np.float32([[1, 0, dx], [0, 1, dy]])
                frame = cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]),
                                       borderMode=cv2.BORDER_REPLICATE)
        return frame
