"""
Event effects: short animations triggered by game events.

  * floating text  - pops in at the player's head (overshoot), rises, fades;
                     in that player's colour ("+10", "READY!") or red ("WRONG",
                     which also wobbles sideways)
  * rings          - a ring in the player's colour expands from their face
  * banners        - centred messages for both players ("GO!", "TIME'S UP");
                     level-ups get a colour band that slides across the screen
  * captions       - the feedback line (feedback.py) under its effect, e.g.
                     "Lightning hands!" under "+10"; pops in just after it

Positions come from anchors: each player's face, else their hands.
"""
import math

import cv2
import numpy as np

import theme

from render.anim import ease_in_cubic, ease_out_back, ease_out_cubic, fade_out, progress
from render.hud import CARD_BOTTOM
from render.primitives import blend_rounded_rect
from render.text import draw as text


def anchors_from_inputs(inputs):
    """{pid: {"center": (x, y), "r": radius}} - where a player's effects appear."""
    out = {}
    for pid, inp in inputs.items():
        if inp.face_box is not None:
            x, y, w, h = inp.face_box
            out[pid] = {"center": (x + w / 2, y + h / 2), "r": 0.9 * max(w, h)}
        elif inp.hands:
            c = np.mean([h[0, :2] for h in inp.hands], axis=0)
            out[pid] = {"center": (float(c[0]), float(c[1])), "r": 80.0}
    return out


class EffectsLayer:
    def __init__(self):
        self.fx = []

    def _add(self, kind, t, dur, pid=None, label="", color=None, size=84, sub=None):
        self.fx.append({"kind": kind, "pid": pid, "t0": t, "dur": dur, "text": label,
                        "color": color, "size": size, "sub": sub})

    def clear(self):
        self.fx = []

    def on_events(self, events, t, lines=None):
        """lines: optional feedback line per event (same order), shown as captions."""
        lines = lines or [None] * len(events)
        game_over = any(e["type"] == "game_over" for e in events)
        for e, line in zip(events, lines):
            k = e["type"]
            if k == "score_event":
                pid = e["player_id"]
                self._add("float", t, 1600, pid, f"+{e['points']}", theme.player_color(pid), 76,
                          sub=line)
                self._add("ring", t, 700, pid, color=theme.player_color(pid))
            elif k == "wrong_sign":
                self._add("float", t, 1400, e["player_id"], "WRONG", theme.BAD, 48, sub=line)
            elif k == "action_event" and e["action"] == "ready":
                pid = e["player_id"]
                self._add("float", t, 1000, pid, "READY!", theme.player_color(pid), 52)
                self._add("ring", t, 600, pid, color=theme.GOOD)
            elif k == "clash":
                self._add("banner", t, 1500, label="CLASH!", color=theme.ACCENT, sub=line)
            elif k == "round_timeout" and not game_over:     # the game-over screen takes over
                self._add("banner", t, 1400, label="TIME'S UP", color=theme.BAD, sub=line)
            elif k == "round_start" and e["first"]:
                self._add("banner", t, 900, label="GO!", color=theme.ACCENT, size=110)
            elif k == "level_up":
                self._add("band", t, 1800, label=f"LEVEL {e['level']}!",
                          color=theme.LEVEL_COLORS.get(e["level"], theme.ACCENT), sub=line)

    def draw(self, frame, t, anchors):
        self.fx = [f for f in self.fx if t - f["t0"] < f["dur"]]
        for f in self.fx:
            u = progress(t, f["t0"], f["dur"])
            if f["kind"] == "banner":
                self._banner(frame, f, u)
            elif f["kind"] == "band":
                self._band(frame, f, u)
            elif f["pid"] in anchors:
                if f["kind"] == "float":
                    self._float(frame, f, u, anchors[f["pid"]])
                else:
                    self._ring(frame, f, u, anchors[f["pid"]])

    @staticmethod
    def _float(frame, f, u, a):
        cx, cy, r = a["center"][0], a["center"][1], a["r"]
        pop = ease_out_back(min(1.0, u / 0.25))                  # first quarter: pop in
        rise = ease_out_cubic(u) * 70
        top = max(cy - r * 0.75 - 20 - rise, CARD_BOTTOM + 70)   # never under the target card
        if f["text"] == "WRONG":                                 # wobble sideways: "no"
            cx += 12 * math.sin(u * 30) * (1 - u)
        text(frame, f["text"], (cx, top), f["size"] * (0.4 + 0.6 * pop), f["color"],
             align="center", alpha=fade_out(u))
        _caption(frame, f, u, cx, top + 40, f["color"] if f["color"] != theme.BAD else theme.WHITE)

    @staticmethod
    def _ring(frame, f, u, a):
        cx, cy, r = int(a["center"][0]), int(a["center"][1]), a["r"]
        radius = int(r * (0.8 + 0.8 * ease_out_cubic(u)))
        thick = max(1, int(8 * (1 - u)))
        cv2.circle(frame, (cx, cy), radius, f["color"], thick, cv2.LINE_AA)

    @staticmethod
    def _banner(frame, f, u):
        H, W = frame.shape[:2]
        pop = ease_out_back(min(1.0, u / 0.3))
        y = H // 2 + 10 - 25 * ease_out_cubic(u)
        text(frame, f["text"], (W // 2, y), f["size"] * (0.5 + 0.5 * pop), f["color"],
             align="center", alpha=fade_out(u, 0.75))
        _caption(frame, f, u, W // 2, y + 56, theme.WHITE, size=40)

    @staticmethod
    def _band(frame, f, u):
        """A colour band slides in from the left, holds, and slides out to the right."""
        H, W = frame.shape[:2]
        if u < 0.2:
            x = -W + W * ease_out_cubic(u / 0.2)
        elif u > 0.8:
            x = W * ease_in_cubic((u - 0.8) / 0.2)
        else:
            x = 0
        y, band_h = H // 2 - 65, 130
        blend_rounded_rect(frame, x, y, W, band_h, f["color"], 0.55, radius=0)
        text(frame, f["text"], (x + W / 2, y + 92), 88, theme.WHITE, align="center")
        _caption(frame, f, u, x + W / 2, y + band_h + 44, theme.WHITE, size=40)


def _caption(frame, f, u, cx, y, color, size=34, delay=0.12):
    """The feedback line under an effect: pops in shortly after it, fades with it."""
    if not f.get("sub"):
        return
    v = (u - delay) / (1 - delay)
    if v <= 0:
        return
    pop = ease_out_back(min(1.0, v / 0.25))
    text(frame, f["sub"], (cx, y), size * (0.5 + 0.5 * pop), color, align="center",
         alpha=fade_out(v, 0.75))
