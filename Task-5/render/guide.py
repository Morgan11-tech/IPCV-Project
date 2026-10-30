"""
Helping new players: the "how to play" panel shown in the lobby.
"""
import cv2

import theme

from render.anim import clamp01, ease_out_back, progress
from render.primitives import panel
from render.text import draw as text

HOW_TO_PLAY = [
    ("Hold an open palm", "both players, to get READY"),
    ("Sign the word on the card", "first correct sign scores +10"),
    ("Wrong sign or time's up", "costs a heart. Last one standing wins!"),
]


def draw_how_to_play(frame, t, shown_t):
    """Lobby panel in the middle of the screen; the three steps pop in one by one."""
    H, W = frame.shape[:2]
    pw, ph = 520, 284
    x, y = W // 2 - pw // 2, int(H * 0.30)
    panel(frame, x, y, pw, ph)
    text(frame, "HOW TO PLAY", (W // 2, y + 44), 38, theme.ACCENT, align="center")
    for i, (title, sub) in enumerate(HOW_TO_PLAY):
        pop = ease_out_back(progress(t, shown_t + 250 * i, 400)) if shown_t is not None else 1.0
        if pop <= 0:
            continue
        row = y + 92 + 56 * i
        cv2.circle(frame, (x + 42, row - 10), int(17 * pop), theme.ACCENT, -1, cv2.LINE_AA)
        text(frame, str(i + 1), (x + 42, row), 24 * pop, theme.PANEL, align="center",
             outline=False)
        text(frame, title, (x + 74, row - 4), 26 * (0.6 + 0.4 * pop), theme.WHITE)
        text(frame, sub, (x + 74, row + 20), 18, theme.WHITE, font="body", alpha=clamp01(0.85 * pop))
    text(frame, "3 levels, less time each level", (W // 2, y + ph - 16), 18, theme.WHITE,
         font="body", align="center", alpha=0.7)
