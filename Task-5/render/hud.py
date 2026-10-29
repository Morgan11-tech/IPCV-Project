"""
HUD: each player's half (colour strip + score panel with hearts and status)
and the target card at the top centre (level, word, timer, countdown,
lobby/game-over). Reads Task 4's snapshot, the per-player inputs and HudAnim
(when things last changed, for the animations); draws nothing else.

Animations: score number bumps when it changes, a lost heart swells and
fades to grey, a new word pops into the card, the word and timer pulse in
the last 3 seconds, and the countdown numbers pop in one per second.
"""
import math

import theme
from task4_game import MAX_HEARTS

from render.anim import bump, ease_out_back, fade_out, progress, pulse
from render.primitives import bar, heart, panel
from render.text import draw as text

PANEL_W, PANEL_H, MARGIN = 380, 108, 20
CARD_W, CARD_H, CARD_Y = 480, 124, 14
CARD_BOTTOM = CARD_Y + CARD_H + 5           # effects stay below this (incl. shadow)
HURRY_S = 3.0                               # last seconds: timer turns red and pulses
HEART_LOSS_MS = 600


class HudAnim:
    """When HUD values last changed (ms), so the HUD can animate the change.
    Score and hearts are compared with the previous frame's snapshot."""

    def __init__(self):
        self.score_t = {}        # pid -> t of last score increase
        self.heart_lost = {}     # pid -> (t, index of the heart that was lost)
        self.word_t = None       # t a new target word appeared
        self._prev = {}          # pid -> (score, hearts) last frame

    def observe(self, snap, events, t):
        for p in snap["players"]:
            pid, score, hearts = p["player_id"], p["score"], p.get("hearts", MAX_HEARTS)
            if pid in self._prev:
                old_score, old_hearts = self._prev[pid]
                if score > old_score:
                    self.score_t[pid] = t
                if hearts < old_hearts:
                    self.heart_lost[pid] = (t, hearts)
            self._prev[pid] = (score, hearts)
        if any(e["type"] == "round_start" for e in events):
            self.word_t = t

    def clear(self):
        self.__init__()


def draw_halves(frame):
    """A strip in each player's colour along the bottom of their half."""
    H, W = frame.shape[:2]
    for pid in theme.PLAYER_IDS:
        x0, x1 = theme.half_bounds(pid, W)
        frame[H - theme.HALF_STRIP:H, x0:x1] = theme.player_color(pid)


def _status(p, snap, inp):
    """(text, colour) for the line under a player's hearts."""
    if snap["phase"] != "playing":
        if not snap.get("rematch_open", True):
            return "", theme.WHITE
        return ("READY!", theme.GOOD) if p["ready"] else ("hold an open palm", theme.WHITE)
    if p["locked_out"]:
        return "LOCKED OUT", theme.BAD
    if inp is not None and inp.word:
        target = snap["target"]["word"] if snap["target"] else ""
        prob = inp.word_probs.get(target, 0.0)
        return f"{inp.word}  {prob:.0%}", theme.GOOD if inp.locked else theme.WHITE
    return "", theme.WHITE


def _draw_hearts(frame, x, y, hearts, color, lost, t):
    for i in range(MAX_HEARTS):
        cx = x + 32 + 34 * i
        if lost is not None and lost[1] == i and t - lost[0] < HEART_LOSS_MS:
            u = progress(t, lost[0], HEART_LOSS_MS)
            heart(frame, cx, y + 60, 26 * (1 + 0.6 * math.sin(math.pi * u)),
                  color if u < 0.5 else theme.EMPTY_HEART)
        else:
            heart(frame, cx, y + 60, 26, color if i < hearts else theme.EMPTY_HEART)


def draw_player_panel(frame, p, snap, inp, anim, t):
    H, W = frame.shape[:2]
    pid = p["player_id"]
    color = theme.player_color(pid)
    x = MARGIN if pid == 1 else W - MARGIN - PANEL_W
    y = H - theme.HALF_STRIP - 16 - PANEL_H
    panel(frame, x, y, PANEL_W, PANEL_H, border=color, border_thick=3)

    text(frame, theme.PLAYER_NAMES[pid], (x + 18, y + 36), 28, color)
    score_size = 56 * bump(t, anim.score_t.get(pid), 350, 0.45)
    text(frame, str(p["score"]), (x + PANEL_W - 18, y + 58), score_size,
         theme.ACCENT if score_size > 57 else theme.WHITE, align="right")
    _draw_hearts(frame, x, y, p.get("hearts", MAX_HEARTS), color, anim.heart_lost.get(pid), t)
    streak = p.get("streak", 0)
    if streak >= 2:                                   # grows with the streak
        text(frame, f"STREAK x{streak}", (x + PANEL_W - 18, y + 96), min(30, 16 + 2 * streak),
             theme.ACCENT, align="right")
    status, status_color = _status(p, snap, inp)
    text(frame, status, (x + 18, y + 96), 20, status_color, font="body")


def _draw_countdown(frame, snap, t, x, y, cx):
    remaining = snap["countdown_ms"]
    n = math.ceil(remaining / 1000)
    u = 1 - (remaining % 1000) / 1000 if remaining % 1000 else 0.0   # 0 -> 1 within each second
    text(frame, "GET READY", (cx, y + 36), 26, theme.WHITE, align="center")
    text(frame, str(n), (cx, y + 108), 72 * (0.5 + 0.5 * ease_out_back(min(1, u / 0.3))),
         theme.ACCENT, align="center", alpha=fade_out(u, 0.8))


def draw_target_card(frame, snap, t, anim):
    W = frame.shape[1]
    x, y = W // 2 - CARD_W // 2, CARD_Y
    cx = W // 2
    phase = snap["phase"]

    if phase == "waiting":
        panel(frame, x, y, CARD_W, 100)
        text(frame, "HOLD AN OPEN PALM", (cx, y + 50), 40 * (1 + 0.04 * pulse(t, 1600)),
             theme.ACCENT, align="center")
        text(frame, "both players, to start", (cx, y + 82), 22, theme.WHITE, font="body",
             align="center")
        return

    if phase == "game_over":
        winner = snap["winner"]
        panel(frame, x, y, CARD_W, CARD_H,
              border=theme.player_color(winner) if winner else None, border_thick=3)
        text(frame, "GAME OVER", (cx, y + 44), 40, theme.WHITE, align="center")
        if winner:
            text(frame, f"{theme.PLAYER_NAMES[winner]} WINS", (cx, y + 82), 32,
                 theme.player_color(winner), align="center")
        else:
            text(frame, "DRAW", (cx, y + 82), 32, theme.ACCENT, align="center")
        if snap.get("rematch_open", True):
            text(frame, "hold an open palm to play again", (cx, y + 110), 18, theme.WHITE,
                 font="body", align="center", alpha=0.6 + 0.4 * pulse(t, 1400))
        return

    level = snap["level"]
    panel(frame, x, y, CARD_W, CARD_H)
    if snap.get("countdown_ms", 0) > 0:
        _draw_countdown(frame, snap, t, x, y, cx)
        return
    text(frame, f"LEVEL {level}", (x + 18, y + 30), 22, theme.LEVEL_COLORS.get(level, theme.WHITE))
    text(frame, "SIGN THIS", (x + CARD_W - 18, y + 30), 18, theme.WHITE, font="body", align="right")
    target = snap["target"]
    if not target:
        return
    secs = max(0.0, (target["expires_ms"] - t) / 1000.0)
    hurry = snap["round_open"] and secs <= HURRY_S
    pop = ease_out_back(progress(t, anim.word_t, 350)) if anim.word_t is not None else 1.0
    size = 62 * (0.3 + 0.7 * pop) * ((1 + 0.08 * pulse(t, 500)) if hurry else 1)
    text(frame, target["word"].upper(), (cx, y + 84), size, theme.ACCENT, align="center")
    if snap["round_open"]:
        span = max(target["expires_ms"] - target["spawned_ms"], 1)
        bar(frame, x + 18, y + 100, CARD_W - 110, 12, (target["expires_ms"] - t) / span,
            theme.BAD if hurry else theme.ACCENT)
        text(frame, f"{secs:.1f}s", (x + CARD_W - 18, y + 112),
             22 * (1 + 0.15 * pulse(t, 500)) if hurry else 22,
             theme.BAD if hurry else theme.WHITE, font="body", align="right")


def draw_hud(frame, snap, inputs, t, anim):
    draw_halves(frame)
    for p in snap["players"]:
        draw_player_panel(frame, p, snap, inputs.get(p["player_id"]), anim, t)
    draw_target_card(frame, snap, t, anim)
