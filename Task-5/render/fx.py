"""
Event effects: short animations triggered by Task 4's events, anchored to the
player who caused them (their face, else their hands) and drawn in that
player's colour. Banners for events that concern both players are centred.
"""
import cv2
import numpy as np

import theme

from render.primitives import text


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

    def _add(self, kind, pid, t, dur, label=""):
        self.fx.append({"kind": kind, "pid": pid, "t0": t, "dur": dur, "text": label})

    def clear(self):
        self.fx = []

    def on_events(self, events, t):
        for e in events:
            k = e["type"]
            if k == "score_event":
                self._add("score", e["player_id"], t, 1300, f"+{e['points']}")
            elif k == "wrong_sign":
                self._add("wrong", e["player_id"], t, 900, "WRONG")
            elif k == "clash":
                self._add("banner", None, t, 1300, "CLASH!")
            elif k == "round_timeout":
                self._add("banner", None, t, 1000, "TIME'S UP")
            elif k == "action_event" and e["action"] == "ready":
                self._add("ready", e["player_id"], t, 1000, "READY!")
            elif k == "game_start":
                self._add("banner", None, t, 1200, "GO!")
            elif k == "level_up":
                self._add("banner", None, t, 1500, f"LEVEL {e['level']}!")

    def draw(self, frame, t, anchors):
        self.fx = [f for f in self.fx if t - f["t0"] < f["dur"]]
        H, W = frame.shape[:2]
        for f in self.fx:
            u = (t - f["t0"]) / f["dur"]           # 0 -> 1 over the effect's life
            if f["kind"] == "banner":
                text(frame, f["text"], (W // 2, H // 2 - 40 - int(30 * u)), 2.0, theme.ACCENT, 4,
                     align="center")
                continue
            a = anchors.get(f["pid"])
            if a is None:
                continue
            color = theme.player_color(f["pid"])
            cx, cy, r = int(a["center"][0]), int(a["center"][1]), int(a["r"])
            top = max(int(cy - r - 25 - 60 * u), 60)
            if f["kind"] == "score":
                cv2.circle(frame, (cx, cy), int(r * (1 + 0.6 * u)), color, 4, cv2.LINE_AA)
                text(frame, f["text"], (cx, top), 1.8, color, 4, align="center")
            elif f["kind"] == "wrong":
                text(frame, f["text"], (cx, top), 1.1, theme.BAD, 3, align="center")
            elif f["kind"] == "ready":
                text(frame, f["text"], (cx, top), 1.2, color, 3, align="center")
