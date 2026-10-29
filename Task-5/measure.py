"""
Task 5 - timing: per-stage milliseconds for one frame, and a smoothed FPS.
"""
import time


class FrameTimer:
    """Lap timer for one frame: start(), then lap("stage") after each stage."""

    def __init__(self):
        self.ms = {}
        self._t0 = self._last = time.perf_counter()

    def start(self):
        self.ms = {}
        self._t0 = self._last = time.perf_counter()
        return self._t0

    def lap(self, name):
        now = time.perf_counter()
        self.ms[name] = (now - self._last) * 1000
        self._last = now

    @property
    def total(self):
        return (self._last - self._t0) * 1000


class FpsMeter:
    """Frames per second of the whole loop, EMA-smoothed so it is readable."""

    def __init__(self, alpha=0.1):
        self.alpha, self.fps, self._last = alpha, 0.0, None

    def tick(self):
        now = time.perf_counter()
        if self._last is not None:
            instant = 1.0 / max(now - self._last, 1e-6)
            self.fps = instant if self.fps == 0 else self.alpha * instant + (1 - self.alpha) * self.fps
        self._last = now
        return self.fps
