"""
Things attached to each player's face: Task 1's face ring (Task 1's effect,
drawn in the player's colour) and the name tag under the chin. The tag
follows the smoothed face box and scales with face size, like Task 1's tag,
but in the game's font and style.
"""
import effects as face_effects          # Task 1
import numpy as np

import theme

from render.primitives import blend_rounded_rect, rounded_rect
from render.text import draw as text, measure


def draw_name_tag(frame, player):
    if not player.visible:
        return
    x, y, w, h = player.smooth_box
    cx = x + w / 2
    size = int(np.clip(w / 5.5, 16, 34))
    label = f"{theme.PLAYER_NAMES[player.player_id]}  {player.score}"
    tw, th = measure(label, size, outline=False)
    pad_x, pad_y = size // 2, size // 3
    top = y + h * 1.18
    x0, x1 = cx - tw / 2 - pad_x, cx + tw / 2 + pad_x
    y0, y1 = top, top + th + 2 * pad_y
    color = theme.player_color(player.player_id)
    blend_rounded_rect(frame, x0, y0 + 3, x1 - x0, y1 - y0, theme.OUTLINE, 0.35, radius=size)
    rounded_rect(frame, x0, y0, x1, y1, color, -1, radius=size)
    text(frame, label, (cx, y1 - pad_y - size * 0.12), size, theme.WHITE, align="center",
         outline=False)


def draw_players(frame, players):
    for player in players.values():
        face_effects.draw_face_effect(frame, player)
        draw_name_tag(frame, player)
