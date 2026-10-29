"""
Operator overlays (debug mode only): stage timings + FPS and the stand-in
Player 2 used when testing alone. (Task 3's hand-ownership view is drawn by
Task 3's own draw_debug.)
"""
import cv2

import theme

from render.primitives import panel
from render.text import draw as text


def draw_stats(frame, fps, timer):
    W = frame.shape[1]
    lines = [f"FPS {fps:5.1f}   frame {timer.total:5.1f} ms"]
    lines += [f"{name:<9}{ms:6.1f} ms" for name, ms in timer.ms.items()]
    x, y = W - 270, 140
    panel(frame, x - 10, y - 24, 260, 22 * len(lines) + 14, alpha=0.6)
    for i, line in enumerate(lines):
        text(frame, line, (x, y + 22 * i), 17, theme.WHITE, font="body", outline=False)


def draw_fake_player(frame, inp):
    x, y, w, h = [int(v) for v in inp.face_box]
    color = theme.player_color(2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
    text(frame, "P2 (fake)", (x, y - 10), 26, color)
    for px, py, _ in inp.hands[0]:
        cv2.circle(frame, (int(px), int(py)), 3, color, -1)
