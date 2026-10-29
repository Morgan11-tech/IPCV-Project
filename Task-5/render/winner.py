"""
Game-over celebration: "<NAME> WINS!" pops in and gently bounces in the
winner's colour, with the final score and a feedback line underneath. The
confetti rain behind it comes from the particle system (render/__init__.py).
"""
import numpy as np

import theme

from render.anim import clamp01, ease_out_back
from render.text import draw as text


class WinnerScreen:
    def __init__(self):
        self.start_t, self.winner, self.score, self.line = None, None, 0, None

    def start(self, t, winner_id, score=0, line=None):
        self.start_t, self.winner, self.score, self.line = t, winner_id, score, line

    def stop(self):
        self.start_t, self.winner = None, None

    def draw(self, frame, t):
        if self.start_t is None:
            return
        H, W = frame.shape[:2]
        c = theme.player_color(self.winner)
        u = (t - self.start_t) / 1000.0
        pop = ease_out_back(min(1.0, u / 0.5))
        y = int(H * 0.45) + int(8 * np.sin(u * 4.0))                         # gentle bounce

        text(frame, f"{theme.PLAYER_NAMES[self.winner]} WINS!", (W // 2, y), 84 * (0.4 + 0.6 * pop),
             c, align="center")
        text(frame, f"Final score: {self.score}", (W // 2, y + 50), 32, theme.WHITE,
             font="body", align="center", alpha=clamp01(u / 0.6))
        if self.line:
            text(frame, self.line, (W // 2, y + 110), 44, theme.ACCENT, align="center",
                 alpha=clamp01((u - 0.4) / 0.5))
