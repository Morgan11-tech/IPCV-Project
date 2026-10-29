"""
HUD: each player's half (colour strip + score panel with hearts and status)
and the target card at the top centre (level, word, timer, lobby/game-over).
Reads only Task 4's snapshot and the per-player inputs; draws nothing else.
"""
import theme
from task4_game import MAX_HEARTS

from render.primitives import bar, heart, panel, text

PANEL_W, PANEL_H, MARGIN = 380, 104, 20
CARD_W, CARD_H = 460, 112


def draw_halves(frame):
    """A strip in each player's colour along the bottom of their half."""
    H, W = frame.shape[:2]
    for pid in theme.PLAYER_IDS:
        x0, x1 = theme.half_bounds(pid, W)
        frame[H - theme.HALF_STRIP:H, x0:x1] = theme.player_color(pid)


def _status(p, snap, inp):
    """(text, colour) for the line under a player's hearts."""
    if snap["phase"] != "playing":
        return ("READY!", theme.GOOD) if p["ready"] else ("hold an open palm", theme.WHITE)
    if p["locked_out"]:
        return "LOCKED OUT", theme.BAD
    if inp is not None and inp.word:
        target = snap["target"]["word"] if snap["target"] else ""
        prob = inp.word_probs.get(target, 0.0)
        return f"{inp.word}  {prob:.0%}", theme.GOOD if inp.locked else theme.WHITE
    return "", theme.WHITE


def draw_player_panel(frame, p, snap, inp):
    H, W = frame.shape[:2]
    pid = p["player_id"]
    color = theme.player_color(pid)
    x = MARGIN if pid == 1 else W - MARGIN - PANEL_W
    y = H - theme.HALF_STRIP - 14 - PANEL_H
    panel(frame, x, y, PANEL_W, PANEL_H, border=color, border_thick=3)

    text(frame, theme.PLAYER_NAMES[pid], (x + 16, y + 32), 0.75, color)
    text(frame, str(p["score"]), (x + PANEL_W - 16, y + 48), 1.5, theme.WHITE, 3, align="right")
    hearts = p.get("hearts", MAX_HEARTS)
    for i in range(MAX_HEARTS):
        heart(frame, x + 30 + 32 * i, y + 58, 24, color if i < hearts else theme.EMPTY_HEART)
    if p.get("streak", 0) >= 2:
        text(frame, f"STREAK x{p['streak']}", (x + PANEL_W - 16, y + 72), 0.55, theme.ACCENT, 1,
             align="right")
    status, status_color = _status(p, snap, inp)
    if status:
        text(frame, status, (x + 16, y + 94), 0.6, status_color, 1)


def draw_target_card(frame, snap, t):
    W = frame.shape[1]
    x, y = W // 2 - CARD_W // 2, 14
    cx = W // 2
    phase = snap["phase"]

    if phase == "waiting":
        panel(frame, x, y, CARD_W, CARD_H - 20)
        text(frame, "HOLD AN OPEN PALM", (cx, y + 42), 1.0, theme.ACCENT, 2, align="center")
        text(frame, "both players, to start", (cx, y + 74), 0.65, theme.WHITE, 1, align="center")
        return

    if phase == "game_over":
        winner = snap["winner"]
        panel(frame, x, y, CARD_W, CARD_H,
              border=theme.player_color(winner) if winner else None, border_thick=3)
        text(frame, "GAME OVER", (cx, y + 38), 1.0, theme.WHITE, 2, align="center")
        if winner:
            text(frame, f"{theme.PLAYER_NAMES[winner]} WINS", (cx, y + 72), 0.9,
                 theme.player_color(winner), 2, align="center")
        else:
            text(frame, "DRAW", (cx, y + 72), 0.9, theme.ACCENT, 2, align="center")
        text(frame, "hold an open palm to play again", (cx, y + 100), 0.55, theme.WHITE, 1,
             align="center")
        return

    level = snap["level"]
    panel(frame, x, y, CARD_W, CARD_H)
    text(frame, f"LEVEL {level}", (x + 16, y + 28), 0.6, theme.LEVEL_COLORS.get(level, theme.WHITE), 1)
    text(frame, "SIGN", (x + CARD_W - 16, y + 28), 0.6, theme.WHITE, 1, align="right")
    target = snap["target"]
    if not target:
        return
    text(frame, target["word"].upper(), (cx, y + 72), 1.5, theme.ACCENT, 3, align="center")
    if snap["round_open"]:
        span = max(target["expires_ms"] - target["spawned_ms"], 1)
        secs = max(0.0, (target["expires_ms"] - t) / 1000.0)
        hurry = secs <= 3
        bar(frame, x + 16, y + 88, CARD_W - 100, 12, (target["expires_ms"] - t) / span,
            theme.BAD if hurry else theme.ACCENT)
        text(frame, f"{secs:4.1f}s", (x + CARD_W - 16, y + 100), 0.6,
             theme.BAD if hurry else theme.WHITE, 1, align="right")


def draw_hud(frame, snap, inputs, t):
    draw_halves(frame)
    for p in snap["players"]:
        draw_player_panel(frame, p, snap, inputs.get(p["player_id"]))
    draw_target_card(frame, snap, t)
