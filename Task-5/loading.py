"""
Task 5 - the loading screen shown at start-up, before registration.

The slow start-up steps (pipeline.loading_steps(): face, hand/sign, pose and
segmentation models) run one by one in a background thread, so the title
screen keeps animating at ~60 fps while they load. The bar shows real
progress: it fills per finished step and creeps forward inside a step that is
still running, then says "Ready!" and fades to black.

The camera is opened on the main thread (macOS only allows that), shown as
the first step "Starting camera".
"""
import math
import threading
import time

import cv2

from render.title import TitleScreen

MIN_SHOW_MS = 2000      # show the title at least this long, even on a fast laptop
READY_HOLD_MS = 500     # "Ready!" with a full bar
FADE_MS = 350           # fade to black into registration
CREEP_MS = 1500         # inside a running step the bar creeps towards its end
SMOOTH = 0.15           # bar follows its target smoothly


class LoadingScreen:
    def __init__(self, window, W, H):
        self.window, self.W, self.H = window, W, H
        self.title = TitleScreen()
        self._start = time.perf_counter()

    def _t(self):
        return (time.perf_counter() - self._start) * 1000

    def show(self, label, frac):
        """Draw a single frame (used around the blocking camera start)."""
        cv2.imshow(self.window, self.title.draw(self.W, self.H, self._t(), frac, label))
        cv2.waitKey(1)

    def run(self, steps, start_frac=0.0, audio=None):
        """Run steps [(label, fn)] in a background thread while animating.
        Returns False if the player quit (Q / Esc), True when done.
        An exception in a step is raised here, on the main thread."""
        state = {"done": 0, "step_t": self._t(), "error": None, "finished": False}

        def work():
            try:
                for _, fn in steps:
                    fn()
                    state["done"] += 1
                    state["step_t"] = self._t()
            except Exception as e:                      # handed to the main thread
                state["error"] = e
            state["finished"] = True

        threading.Thread(target=work, daemon=True).start()
        n, shown, ready_t = len(steps), start_frac, None
        while True:
            t = self._t()
            if state["error"] is not None:
                raise state["error"]
            done = state["done"]
            creep = 0.0 if done >= n else 0.9 * (1 - math.exp(-(t - state["step_t"]) / CREEP_MS))
            target = start_frac + (1 - start_frac) * min(1.0, (done + creep) / n)
            shown += (target - shown) * SMOOTH
            label = "Ready!" if state["finished"] else steps[min(done, n - 1)][0] + "..."

            if state["finished"] and t >= MIN_SHOW_MS and shown > 0.995 and ready_t is None:
                ready_t = t
            frame = self.title.draw(self.W, self.H, t, 1.0 if ready_t else shown, label)
            if ready_t is not None:
                fade = (t - ready_t - READY_HOLD_MS) / FADE_MS
                if fade >= 1:
                    return True
                if fade > 0:
                    frame = cv2.convertScaleAbs(frame, alpha=1 - fade)
            if audio is not None:
                audio.update({"phase": "waiting", "level": 1})
            cv2.imshow(self.window, frame)
            if cv2.waitKey(16) & 0xFF in (ord("q"), 27):
                return False
